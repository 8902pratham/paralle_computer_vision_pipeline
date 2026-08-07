import os
import time
import shutil
import cv2
import pandas as pd
from dask import delayed, compute
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    output_path = os.path.join(OUTPUT_FOLDER, image_filename)
    cv2.imwrite(output_path, final_img)
    return output_path

# === Chunk Utility ===
def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# === Process a Chunk ===
def process_chunk(chunk):
    return [clahe(f) for f in chunk]

# === Benchmarking ===
def performance(cpus, image_files):
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

    client = Client(n_workers=cpus, threads_per_worker=THREADS_PER_WORKER, processes=True)
    print(f"\nRunning with {cpus} CPUs...")

    chunks = list(chunk_list(image_files, CHUNK_SIZE))
    tasks = [delayed(process_chunk)(chunk) for chunk in chunks]

    start_time = time.time()
    with performance_report(filename=f"dask_report_{cpus}_cpus.html"):
        compute(*tasks)
    end_time = time.time()
    client.close()

    time_taken = round(end_time - start_time, 2)
    print(f"Completed in {time_taken} seconds.")
    return time_taken

# === Main ===
if __name__ == "__main__":
    valid_exts = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in os.listdir(INPUT_FOLDER)
                   if os.path.isfile(os.path.join(INPUT_FOLDER, f)) and os.path.splitext(f)[1].lower() in valid_exts]

    results = []

    for i, cpus in enumerate(cpu_counts):
        time_taken = performance(cpus, image_files)
        speedup = round(results[0]['Time (s)'] / time_taken, 2) if i > 0 else 1.0
        efficiency = round(speedup / cpus, 2)
        results.append({
            "CPUs": cpus,
            "Time (s)": time_taken,
            "Speedup": speedup,
            "Efficiency": efficiency,
            "Images": len(image_files)
        })

    df = pd.DataFrame(results)
    print("\n=== Performance Summary ===")
    print(df.to_string(index=False))
