import os
import time
import shutil
import cv2
import pandas as pd
from joblib import Parallel, delayed
from dask import delayed as dask_delayed, compute
from dask.distributed import Client, performance_report

# === Configuration ===
INPUT_FOLDER = "/courses/CSYE7105.202530/students/kalakoti.v/data/images/train"
OUTPUT_FOLDER = "images/val"
cpu_counts = [12, 18, 36]
THREADS_PER_WORKER = 1
CHUNK_SIZE = 100

# === CLAHE Processing ===
def clahe(image_filename):
    input_path = os.path.join(INPUT_FOLDER, image_filename)
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe_op.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    output_path = os.path.join(OUTPUT_FOLDER, image_filename)
    cv2.imwrite(output_path, final_img)
    return output_path

# === Chunk Utility ===
def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# === Dask Benchmark ===
def run_dask(cpus, image_files):
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    client = Client(n_workers=cpus, threads_per_worker=THREADS_PER_WORKER, processes=True)
    print(f"\n[DASK] Running with {cpus} CPUs on {len(image_files)} images...")

    chunks = list(chunk_list(image_files, CHUNK_SIZE))
    tasks = [dask_delayed(lambda c: [clahe(f) for f in c])(chunk) for chunk in chunks]

    start_time = time.time()
    with performance_report(filename=f"dask_report_{cpus}_cpus.html"):
        compute(*tasks)
    end_time = time.time()
    client.close()

    duration = round(end_time - start_time, 2)
    print(f"[DASK] Time taken: {duration} seconds")
    return duration

# === Joblib Benchmark ===
def run_joblib(cpus, image_files):
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    print(f"\n[JOBLIB] Running with {cpus} CPUs on {len(image_files)} images...")
    start_time = time.time()
    Parallel(n_jobs=cpus)(delayed(clahe)(img) for img in image_files)
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"[JOBLIB] Time taken: {duration} seconds")
    return duration

# === Main Execution ===
if __name__ == "__main__":
    valid_exts = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in os.listdir(INPUT_FOLDER)
                   if os.path.isfile(os.path.join(INPUT_FOLDER, f)) and os.path.splitext(f)[1].lower() in valid_exts]

    dask_results = []
    joblib_results = []

    for i, cpus in enumerate(cpu_counts):
        # Run Dask
        time_dask = run_dask(cpus, image_files)
        speedup_dask = round(dask_results[0]['Time (s)'] / time_dask, 2) if i > 0 else 1.0
        efficiency_dask = round(speedup_dask / cpus, 2)
        dask_results.append({
            "Method": "Dask",
            "CPUs": cpus,
            "Time (s)": time_dask,
            "Speedup": speedup_dask,
            "Efficiency": efficiency_dask,
            "Images": len(image_files)
        })

        # Run Joblib
        time_joblib = run_joblib(cpus, image_files)
        speedup_joblib = round(joblib_results[0]['Time (s)'] / time_joblib, 2) if i > 0 else 1.0
        efficiency_joblib = round(speedup_joblib / cpus, 2)
        joblib_results.append({
            "Method": "Joblib",
            "CPUs": cpus,
            "Time (s)": time_joblib,
            "Speedup": speedup_joblib,
            "Efficiency": efficiency_joblib,
            "Images": len(image_files)
        })

    # Combine and display results
    all_results = pd.DataFrame(dask_results + joblib_results)
    print("\n=== Full Dataset Performance Summary ===")
    print(all_results.to_string(index=False))
