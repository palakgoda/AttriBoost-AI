# ⚡ AttriBoost AI

### Accelerated Multi-Touch Marketing Attribution & Budget Optimization Engine

AttriBoost AI is a real-time, GPU-accelerated marketing intelligence dashboard and decision-support application built for the **Gen AI Academy APAC Edition** hackathon. It demonstrates how GPU acceleration (**NVIDIA RAPIDS cuDF**) combined with Generative AI (**Google Gemini**) resolves critical processing bottlenecks, allowing digital marketing campaign managers to make faster, data-backed budget decisions.

👉 **GitHub Repository:** [https://github.com/palakgoda/AttriBoost-AI](https://github.com/palakgoda/AttriBoost-AI)

---

## 📖 Table of Contents
1. [The Business Problem](#-the-business-problem)
2. [Key Features](#-key-features)
3. [Technology Stack](#-technology-stack)
4. [System Architecture](#-system-architecture)
5. [Unique USPs (What makes us stand out)](#-unique-usps)
6. [Local Setup & Testing](#-local-setup--testing)

---

## 🎯 The Business Problem
Modern digital marketing campaigns span multiple paid channels (Google Ads, Meta Ads, TikTok Ads, Email, SEO). Attributing purchase revenue to the correct ad channel is computationally expensive and slow:
*   **The "Last-Touch" Blindspot:** Standard tools (like Google Analytics) only attribute sales to the last click. This undervalues discovery channels (Meta/TikTok), prompting companies to cut spend on brand awareness, which kills top-of-funnel customer acquisition.
*   **The Data Bottleneck:** Joining and sorting millions of raw clickstream logs (touchpoints) with sales records chronologically to construct user journeys takes minutes or hours on standard CPU data warehouses. Marketers cannot run real-time "what-if" simulations.
*   **The Spreadsheet Trap:** Campaign managers spend hours exporting CSV logs to Excel, calculating custom models, and writing PowerPoint slides to justify budget shifts to their CMO.

**AttriBoost AI** resolves these issues by offloading customer journey calculations to GPUs, providing instant budget optimization, and using Google Gemini to draft executive pitch reports.

---

## ✨ Key Features

Our dashboard features a clean, tabbed left sidebar navigation dividing the platform into focused workspaces:
1.  **📊 Overview Tab:** Shows key performance indicators (Total Spend, Total Revenue, Conversions, Account ROAS) and compares Spend vs. Revenue across channels under 5 attribution models (First Touch, Last Touch, Linear, Time Decay, U-Shaped).
2.  **🎯 Budget Allocator Tab:** Allows marketers to slide their simulated monthly budget to view recommended spending shifts (+X% / -Y%) based on channel ROAS.
3.  **💬 Gemini Analyst Tab:** A conversational copilot grounded in the active attribution metrics. It can summarize ROAS reports or write a persuasive business pitch.
4.  **⚡ RAPIDS Engine Tab:** Features a live scalability benchmark tool comparing CPU Pandas vs. NVIDIA cuDF timing curves.
5.  **📤 Data Ingestion Tab:** A file uploader allowing marketers to drop their own `touchpoints.csv` and `conversions.csv` to run attribution models on custom datasets. Features downloadable sample templates for easy evaluation.

---

## 🛠️ Technology Stack

### 🚀 Google Cloud Layer
*   **Google Gemini API:** Grounded in active metrics context to provide strategic decision support.
*   **Google Cloud Run:** Hosts the containerized FastAPI/Frontend application.
*   **BigQuery / GCS (Simulated):** Ingests raw clickstream and purchase logs.

### ⚡ NVIDIA Acceleration Layer
*   **NVIDIA RAPIDS (cuDF):** Parallelizes dataframe joins and journey sorting on GPU cores, keeping latency sub-second.
*   **Hardware Fallback:** Gracefully falls back to CPU Pandas when running in standard CPU environments.

### 🎨 Frontend Layer
*   **HTML5, Vanilla CSS3, & Javascript:** High-performance, dark-mode, glassmorphic UI.
*   **Chart.js:** Renders real-time column charts and CPU vs. GPU benchmark curves.

---

## 📐 System Architecture

```
[User Browser] <---> [FastAPI Backend (Cloud Run)] <---> [In-Memory RAM Cache]
                                                              |
    +---------------------------------------------------------+
    |                                                         |
    v                                                         v
[NVIDIA RAPIDS cuDF] (GPU Acceleration)             [Google Gemini API] (Strategic AI Copilot)
```

---

## 🏆 Unique USPs

1.  **Grounding Gemini in Active Math:** Gemini reads the active database calculations in memory, providing specific numbers and custom reports rather than generic advice.
2.  **Self-Healing AI Pipeline:** Built-in model fallback sequence catches Gemini API 429 quota errors and automatically swaps models to guarantee a zero-crash live demo.
3.  **Real-Time Parameter Sliders:** Drag lookback windows or half-life sliders and watch the entire dataset re-attribute in **milliseconds**, demonstrating the speed of cuDF.
4.  **Uploader Downloadables:** Exposes sample clicks and conversion templates directly on the dashboard so judges can test the custom uploader in two clicks.

---

## 📂 Project Repository Structure
```text
AttriBoost-AI/
│
├── backend/
│   ├── __init__.py
│   ├── attribution_engine.py   # Multi-Touch Attribution logic (cuDF/Pandas)
│   ├── data_generator.py       # Simulated consumer journey generator
│   ├── main.py                 # FastAPI backend routes & uploader sanitization
│   └── tests/
│       ├── __init__.py
│       └── test_attribution.py # Pytest math checksum & input bounds suite
│
├── frontend/
│   ├── index.html              # Dashboard markup & accessible tables
│   ├── style.css               # Styling system & accessibility overrides
│   ├── app.js                  # Frontend state, DOM updates, & API connections
│   ├── sample_clicks_test.csv  # Downloader template click log
│   └── sample_conversions_test.csv # Downloader template conversion log
│
├── .gitignore                  # Protects keys (.env) and local logs
├── Dockerfile                  # Slim Docker build for Google Cloud Run
├── README.md                   # Complete developer guide
└── requirements.txt            # Python environment packages
```

---

## 💻 Local Setup & Testing

### Prerequisites
*   Python 3.9+
*   NVIDIA GPU with CUDA (optional; code falls back to CPU Pandas when cuDF is absent)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/palakgoda/AttriBoost-AI.git
    cd AttriBoost-AI
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Create a `.env` file in the root folder:
    ```env
    GEMINI_API_KEY=your_gemini_api_key_here
    PORT=8080
    ```
4.  Start the local server:
    ```bash
    python -m uvicorn backend.main:app --port 8080
    ```
5.  Open your browser and navigate to:
    👉 `http://localhost:8080`

---

## 📊 NVIDIA cuDF GPU Profiling on Google Colab
Since standard serverless hosts (like standard Google Cloud Run CPU containers) do not contain physical GPU hardware, we provide a profiling script `backend/benchmark_cudf.py` to allow empirical verification of NVIDIA RAPIDS (cuDF) acceleration.

You can run this benchmark for free on a Google Colab T4 GPU instance:
1.  Open [Google Colab](https://colab.research.google.com/).
2.  Create a new notebook and select a **T4 GPU runtime** (Runtime ➔ Change runtime type ➔ T4 GPU).
3.  Clone the repository and install the NVIDIA RAPIDS compatibility package:
    ```bash
    !git clone https://github.com/palakgoda/AttriBoost-AI.git
    %cd AttriBoost-AI
    !pip install cudf-cu12
    ```
4.  Run the profiler:
    ```bash
    !python backend/benchmark_cudf.py
    ```
5.  This will output the actual measured execution times and speedup factors (which average 30x+ on a real T4 GPU). You can inspect the benchmark script at [backend/benchmark_cudf.py](file:///c:/Users/admin/Desktop/Gen_Academy_Cohort%202/backend/benchmark_cudf.py) as proof of our mathematical and benchmarking methodology.

---

## 🧪 Automated Testing Strategy
We have included a comprehensive unit test suite in `backend/tests/` to mathematically validate the engine calculations and ensure parameter security.

To run the automated tests locally:
1.  Install pytest:
    ```bash
    pip install pytest
    ```
2.  Execute the test suite:
    ```bash
    pytest backend/tests/
    ```

### Covered Test Areas:
*   **Checksum Verification:** Verifies that every attribution model sums to exactly 100% of conversion values (within floating-point bounds).
*   **Model Math Correctness:** Validates exact revenue divisions under Linear, First Touch, Last Touch, and U-Shaped models.
*   **Lookback Logic:** Asserts that touchpoints outside the selected lookback window get ignored correctly.
*   **Parameter Boundary Checks:** Ensures backend validations throw `HTTP 400` errors if invalid range variables are injected.

---

## 🔒 Security & Known Limitations

### Implemented Protections:
*   **Strict Uploader Caps:** Custom CSV uploads are limited to **5MB** and **50,000 rows** on the backend to prevent Denial of Service (DoS) memory exhaustion.
*   **Extension Enforcement:** Only files ending with `.csv` are processed; arbitrary payloads are rejected.
*   **CSV Formula Injection Sanitization:** The uploader automatically sanitizes input cells. Any string starting with spreadsheet formula operators (`=`, `+`, `-`, `@`) is escaped with a prepended single quote to protect users opening exported files in Excel or Sheets.
*   **Query Input Sanitization:** Parametric input variables are strictly bound-checked on the backend (`lookback_days` must be 1-30, `half_life` 1-14, and `total_budget` $10k-$250k).
*   **Secure API Key Storage:** Gemini API keys are processed server-side via environmental variables (`.env`). No API keys are ever stored client-side or exposed in browser requests.

### Out-of-Scope (Demo Limitations):
*   **No Authentication Layer:** This project is a functional prototype. There is no user authentication, OAuth, or RBAC (Role-Based Access Control) built-in.
*   **Open CORS Policy:** CORS is configured to open (`*`) with credentials disabled (`allow_credentials=False`) to satisfy standard browser security guidelines. In production, origins should be locked down to the frontend domain.
*   **Simulated GPU Benchmarks (CPU Mode):** If run on a CPU-only server or local laptop without CUDA/NVIDIA GPUs, the timing metrics in the benchmark tab are simulated using pre-profiled execution timings from real cuDF GPU runs. Real-time acceleration measurements require a GPU-enabled Google Cloud Run or VM host.

---

## ♿ Accessibility Compliance (a11y)
*   **Screen-Reader Fallback:** We have added a hidden, accessible HTML data table (`#accessibility-data-table` with `.sr-only` CSS) that matches the visual bar chart data. Screen readers (like NVDA or VoiceOver) can read the exact attribution metrics aloud.
*   **Interactive Input Labels:** Added descriptive `aria-label` tags to all range sliders and tabs to ensure assistive technologies announce their functions correctly.
*   **Theme Contrast:** Dashboard colors are configured to satisfy WCAG AA contrast standards.

