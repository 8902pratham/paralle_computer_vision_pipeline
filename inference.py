import torch
import os
import time
from ultralytics import YOLOWorld
import multiprocessing
import gc  

#=====Load the YOLO-World model=====
model = YOLOWorld("train_results_singlegpu/exp_device_0_amp_False/weights/best.pt")
device = torch.device("cuda:0")  # Use GPU 0

# =====Function to run inference on a single image using GPU=====
def infer_image_on_gpu(image_path, save_dir):
    try:   
        torch.cuda.set_device(device)
        print(f"Processing image {image_path} on GPU {device}")

        results = model.predict(image_path)

        result_img_path = os.path.join(save_dir, os.path.basename(image_path))
        results[0].save(result_img_path)
        
        print(f"Saved result to {result_img_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    finally:
        # Free up CUDA memory
        torch.cuda.empty_cache()
        gc.collect()

# =====Function to run inference in parallel using multiprocessing=====
def run_inference_parallel(image_paths, save_dir, num_workers=4):
    start_time = time.time()  # Start wall-clock time

    # Use spawn context to prevent CUDA initialization issues
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(infer_image_on_gpu, [(image_path, save_dir) for image_path in image_paths])
    
    end_time = time.time() 
    return end_time - start_time 

# =====Function to run inference sequentially=====
def run_inference_sequential(image_paths, save_dir):
    start_time = time.time() 

    
    for image_path in image_paths:
        infer_image_on_gpu(image_path, save_dir)
    
    end_time = time.time() 
    return end_time - start_time  

if __name__ == '__main__':
    image_dir = "testing_images"
    
 
    save_dir = "runs/detect/predict_3"
    os.makedirs(save_dir, exist_ok=True)

    #
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg")]

    
    if not image_paths:
        print("No images found in the directory.")
        exit()

    # Number of processes for parallel inference
    num_workers = 3 

    # Run sequential inference
    print("Running sequential inference...")
    sequential_time = run_inference_sequential(image_paths, save_dir)
    print(f"Sequential Inference Time: {sequential_time:.2f} seconds")

    # Run multiprocessing inference
    print(f"Running parallel inference with {num_workers} workers...")
    parallel_time = run_inference_parallel(image_paths, save_dir, num_workers)
    print(f"Parallel Inference Time with {num_workers} workers: {parallel_time:.2f} seconds")

    # Performance metrics
    speedup = sequential_time / parallel_time
    efficiency = speedup / num_workers

    print(f"Speedup: {speedup:.2f}")
    print(f"Efficiency: {efficiency:.2f}")
