import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Body, Header, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import google.generativeai as genai

# Load local environment variables
load_dotenv()

from backend.data_generator import generate_marketing_data
from backend.attribution_engine import (
    run_attribution_pipeline,
    compute_all_models,
    run_performance_benchmark,
    GPU_AVAILABLE
)

app = FastAPI(
    title="AttriBoost AI API",
    description="Accelerated Multi-Touch Attribution & Budget Optimization Engine",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
TP_PATH = os.path.join(DATA_DIR, "touchpoints.csv")
CONV_PATH = os.path.join(DATA_DIR, "conversions.csv")

# Global Cache to avoid costly disk I/O on every API request
_cached_tp = None
_cached_conv = None

def load_data(force_reload=False):
    global _cached_tp, _cached_conv
    if _cached_tp is not None and _cached_conv is not None and not force_reload:
        return _cached_tp, _cached_conv
        
    if not os.path.exists(TP_PATH) or not os.path.exists(CONV_PATH):
        # Trigger generation of a responsive default dataset on demand
        generate_marketing_data(output_dir=DATA_DIR, num_users=20000, total_touchpoints=200000)
    
    print("Loading data from disk into memory cache...")
    _cached_tp = pd.read_csv(TP_PATH)
    _cached_conv = pd.read_csv(CONV_PATH)
    return _cached_tp, _cached_conv

@app.on_event("startup")
def startup_event():
    """Ensure data is ready when the server starts."""
    print("Checking database files...")
    try:
        df_tp, df_conv = load_data()
        print(f"Server ready. Loaded {len(df_tp)} touchpoints and {len(df_conv)} conversions.")
    except Exception as e:
        print(f"Error on startup data generation: {e}")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "gpu_available": GPU_AVAILABLE,
        "environment": "GPU (RAPIDS Active)" if GPU_AVAILABLE else "CPU Fallback"
    }

@app.post("/api/regenerate-data")
def regenerate_dataset(payload: dict = Body(...)):
    """Allows triggering larger dataset generation for scaling tests."""
    num_users = payload.get("num_users", 20000)
    total_touchpoints = payload.get("total_touchpoints", 200000)
    
    try:
        generate_marketing_data(
            output_dir=DATA_DIR,
            num_users=num_users,
            total_touchpoints=total_touchpoints
        )
        # Reload cache with newly generated data
        load_data(force_reload=True)
        return {"status": "success", "message": f"Generated {total_touchpoints} touchpoints."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate data: {str(e)}")

@app.get("/api/summary")
def get_summary_metrics():
    """Returns general overview statistics of the marketing dataset."""
    try:
        df_tp, df_conv = load_data()
        
        total_spend = float(df_tp['cost'].sum())
        total_conversions = len(df_conv)
        total_revenue = float(df_conv['conversion_value'].sum())
        
        overall_roas = round(total_revenue / total_spend, 2) if total_spend > 0 else 0.0
        unique_converting_users = df_conv['user_id'].nunique()
        total_users = df_tp['user_id'].nunique()
        
        conversion_rate = round((unique_converting_users / total_users) * 100, 2) if total_users > 0 else 0.0
        
        return {
            "total_spend": round(total_spend, 2),
            "total_conversions": total_conversions,
            "total_revenue": round(total_revenue, 2),
            "overall_roas": overall_roas,
            "conversion_rate": conversion_rate,
            "total_touchpoints": len(df_tp)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attribution")
def get_attribution_results(lookback_days: float = 14.0, half_life: float = 7.0):
    """Returns channel metrics for all attribution models."""
    try:
        df_tp, df_conv = load_data()
        all_results = compute_all_models(df_tp, df_conv, half_life=half_life, lookback_days=lookback_days)
        return all_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reallocate")
def reallocate_budget(payload: dict = Body(...)):
    """
    Optimizes a budget based on ROAS of a specific attribution model.
    Applies ROAS-proportional allocation with a 5% security minimum.
    """
    model_name = payload.get("model", "Linear")
    total_budget = payload.get("total_budget", 10000.0)
    lookback_days = payload.get("lookback_days", 14.0)
    half_life = payload.get("half_life", 7.0)
    
    try:
        df_tp, df_conv = load_data()
        all_results = compute_all_models(df_tp, df_conv, half_life=half_life, lookback_days=lookback_days)
        
        if model_name not in all_results:
            raise HTTPException(status_code=400, detail="Invalid attribution model name.")
            
        model_data = all_results[model_name]
        
        # We only allocate to paid channels (Organic Search has 0 spend)
        paid_channels = [item for item in model_data if item['channel'] != 'Organic Search']
        organic_channels = [item for item in model_data if item['channel'] == 'Organic Search']
        
        total_roas = sum(item['roas'] for item in paid_channels)
        
        recommendations = []
        
        # Allocate organic first (always 0 spend)
        for item in organic_channels:
            recommendations.append({
                "channel": item['channel'],
                "current_spend": item['spend'],
                "roas": item['roas'],
                "recommended_spend": 0.0,
                "percentage_change": 0.0
            })
            
        if total_roas == 0:
            # Split equally if no channel has conversions
            share = total_budget / len(paid_channels)
            for item in paid_channels:
                recommendations.append({
                    "channel": item['channel'],
                    "current_spend": item['spend'],
                    "roas": item['roas'],
                    "recommended_spend": round(share, 2),
                    "percentage_change": round(((share - item['spend']) / item['spend'] * 100), 2) if item['spend'] > 0 else 100.0
                })
        else:
            # Spend reallocation logic
            # Maintain a 5% floor budget for active channels to ensure continuous data gathering
            min_floor = total_budget * 0.05
            remaining_budget = total_budget - (min_floor * len(paid_channels))
            
            for item in paid_channels:
                # Share of the ROAS-dependent pool
                roas_share = item['roas'] / total_roas
                recommended = min_floor + (remaining_budget * roas_share)
                
                percentage_change = 0.0
                if item['spend'] > 0:
                    percentage_change = round(((recommended - item['spend']) / item['spend']) * 100, 1)
                else:
                    percentage_change = 100.0
                    
                recommendations.append({
                    "channel": item['channel'],
                    "current_spend": item['spend'],
                    "roas": item['roas'],
                    "recommended_spend": round(recommended, 2),
                    "percentage_change": percentage_change
                })
                
        return {
            "model": model_name,
            "total_budget": total_budget,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmark")
def get_benchmark():
    """Runs a performance scalability test (CPU vs GPU)."""
    try:
        df_tp, df_conv = load_data()
        results = run_performance_benchmark(df_tp, df_conv)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_copilot(
    payload: dict = Body(...),
    x_gemini_key: str = Header(None)
):
    """Conversational marketing analyst powered by Google Gemini."""
    user_message = payload.get("message", "")
    model_name = payload.get("model", "Linear")
    lookback_days = payload.get("lookback_days", 14.0)
    half_life = payload.get("half_life", 7.0)
    
    # Retrieve API key
    gemini_key = x_gemini_key or os.getenv("GEMINI_API_KEY")
    if not gemini_key or gemini_key == "your_gemini_api_key_here" or gemini_key.strip() == "":
        return {
            "reply": "⚠️ **Gemini API Key is missing.**\n\nTo talk to the AI Analyst Copilot, please enter your Gemini API Key in the **Settings** gear panel in the upper-right corner of the dashboard, or set it as `GEMINI_API_KEY` in your deployment environment."
        }
        
    try:
        df_tp, df_conv = load_data()
        all_results = compute_all_models(df_tp, df_conv, half_life=half_life, lookback_days=lookback_days)
        
        # Build context from actual data
        summary_info = []
        for item in all_results.get(model_name, []):
            summary_info.append(
                f"- {item['channel']}: Spend=${item['spend']:,}, Attributed Rev=${item['attributed_revenue']:,}, Conversions={item['attributed_conversions']}, ROAS={item['roas']}"
            )
        metrics_summary = "\n".join(summary_info)
        
        total_spend = df_tp['cost'].sum()
        total_rev = df_conv['conversion_value'].sum()
        overall_roas = total_rev / total_spend if total_spend > 0 else 0.0
        
        # Configure Gemini SDK
        genai.configure(api_key=gemini_key)
        
        # System instructions incorporated in the prompt
        system_instruction = f"""
You are the AttriBoost AI Copilot, an expert digital marketing analyst and data scientist.
The user is viewing their marketing attribution dashboard. Here is the aggregated performance data:

[General Stats]
- Total Marketing Spend: ${total_spend:,.2f}
- Total Generated Revenue: ${total_rev:,.2f}
- Overall Account ROAS: {overall_roas:.2f}

[Attribution Model Selected: {model_name}]
This model distributes conversion credit across user pathways. Here are the channel results under this model:
{metrics_summary}

Answer the user's questions with specific, professional, and highly actionable marketing recommendations.
Suggest where they should reallocate budget (e.g. increase TikTok Ads if its U-Shaped ROAS is high, or scale back Meta Ads if its First Touch is low).
Keep your response concise, bulleted where appropriate, and formatted in clean Markdown. Do not hallucinate metrics.
"""
        # Resilient Model Fallback to handle rate limits (429) or quota constraints
        models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-flash-latest"]
        response = None
        last_error_msg = ""
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    contents=[
                        {"role": "user", "parts": [system_instruction + f"\nUser question: {user_message}"]}
                    ]
                )
                break # Success!
            except Exception as e:
                last_error_msg = str(e)
                print(f"Model {model_name} failed: {e}")
                continue
                
        if response is None:
            raise Exception(last_error_msg or "All generative models failed.")
        
        return {"reply": response.text}
        
    except Exception as e:
        return {
            "reply": f"⚠️ **Error communicating with Gemini API:**\n\n```\n{str(e)}\n```\nDouble check that your API key is correct and active."
        }

@app.post("/api/upload-data")
async def upload_custom_data(
    touchpoints_file: UploadFile = File(...),
    conversions_file: UploadFile = File(...)
):
    """Allows uploading custom CSV files to run attribution on custom data."""
    global _cached_tp, _cached_conv
    try:
        # Load files into DataFrames
        df_tp = pd.read_csv(touchpoints_file.file)
        df_conv = pd.read_csv(conversions_file.file)
        
        # Verify required headers
        req_tp = {'user_id', 'timestamp', 'channel', 'campaign', 'cost'}
        req_conv = {'user_id', 'timestamp', 'conversion_value'}
        
        if not req_tp.issubset(df_tp.columns):
            missing = req_tp - set(df_tp.columns)
            raise HTTPException(status_code=400, detail=f"Touchpoints CSV missing headers: {missing}")
            
        if not req_conv.issubset(df_conv.columns):
            missing = req_conv - set(df_conv.columns)
            raise HTTPException(status_code=400, detail=f"Conversions CSV missing headers: {missing}")
            
        # Ensure timestamp strings are parsed correctly
        df_tp['timestamp'] = pd.to_datetime(df_tp['timestamp'])
        df_conv['timestamp'] = pd.to_datetime(df_conv['timestamp'])
        
        # Save to local CSV path
        df_tp.to_csv(TP_PATH, index=False)
        df_conv.to_csv(CONV_PATH, index=False)
        
        # Force cache reload
        load_data(force_reload=True)
        
        return {
            "status": "success",
            "message": f"Successfully loaded custom dataset with {len(df_tp)} touchpoints and {len(df_conv)} conversions."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing uploaded CSVs: {str(e)}")

# Mount static frontend files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

@app.get("/{path:path}")
def read_static(path: str):
    file_path = os.path.join("frontend", path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return FileResponse("frontend/index.html")
