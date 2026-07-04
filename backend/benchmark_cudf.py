import os
import sys
import time
import pandas as pd

# Add the project root to python path so we can run from anywhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cudf
    GPU_AVAILABLE = True
    print("SUCCESS: NVIDIA cuDF is installed and CUDA GPU is available!")
except ImportError:
    GPU_AVAILABLE = False
    print("NOTICE: NVIDIA cuDF not found. Running in CPU fallback mode.")

from backend.attribution_engine import run_performance_benchmark
from backend.data_generator import generate_marketing_data

def main():
    print("--------------------------------------------------")
    print("  AttriBoost AI - NVIDIA cuDF Performance Profiler ")
    print("--------------------------------------------------")
    
    # 1. Ensure directories exist
    os.makedirs("data", exist_ok=True)
    
    # 2. Generate baseline benchmark dataset (200k touchpoints)
    print("\n[1/3] Generating baseline profiling dataset...")
    generate_marketing_data("data", num_users=20000, total_touchpoints=200000)
    
    # 3. Load datasets
    print("[2/3] Loading data files into memory...")
    df_tp = pd.read_csv("data/touchpoints.csv")
    df_conv = pd.read_csv("data/conversions.csv")
    
    # 4. Execute Benchmark
    print("[3/3] Running execution time benchmarks...\n")
    start_total = time.time()
    results = run_performance_benchmark(df_tp, df_conv)
    end_total = time.time()
    
    # 5. Output Summary Results
    print("==================================================")
    print("             PROFILED BENCHMARK METRICS           ")
    print("==================================================")
    print(f"Device Mode: {'NVIDIA GPU (Active)' if GPU_AVAILABLE else 'CPU Fallback'}")
    print("--------------------------------------------------")
    
    for r in results:
        print(f"Scale Scale: {r['dataset_percentage']}%")
        print(f"Rows Processed: {r['rows_processed']:,}")
        print(f"  - CPU Pandas: {r['cpu_time_ms']:.2f} ms")
        print(f"  - GPU cuDF:   {r['gpu_time_ms']:.2f} ms")
        print(f"  - Speedup:    {r['speedup']:.1f}x")
        print(f"  - Mode Info:  {r['gpu_mode']}")
        print("--------------------------------------------------")
        
    print(f"Total profiling job completed in {end_total - start_total:.2f} seconds.")
    print("==================================================")

if __name__ == "__main__":
    main()
