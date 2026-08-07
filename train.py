import torch
from ultralytics import YOLO
import time
import os

# =====Function to train YOLO model=====
def train_yolo(device_id, amp, model_name, data_yaml, epochs, batch_size, imgsz):
    # Set device
    device = f"cuda:{device_id}"
    print(f"Training on device: {device}, AMP: {amp}")

    # Load model
    model = YOLO(model_name)

    # =====Setting to limit for one gpu=====
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    
    start_time = time.time() # Measure training time
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        amp=amp,
        project="runs/train",
        name=f"exp_device_{device_id}_amp_{amp}",
        exist_ok=True
    )
    end_time = time.time()

    # =====Calculate time taken=====
    time_taken = end_time - start_time
    avg_time_per_epoch = time_taken / epochs

    return time_taken, avg_time_per_epoch

def main():
    # =====Parameters=====
    model_name = "yolov8s-worldv2.pt"  
    data_yaml = "coco.yaml"  
    epochs = 10  
    batch_size = 16  
    imgsz = 640  

    # =====Run training on GPU 0 with AMP=False=====
    time_taken_0, avg_time_0 = train_yolo(
        device_id=[0,1],
        amp=False,
        model_name=model_name,
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz
    )
    print(f"GPU 0 (AMP=False): Total time = {time_taken_0:.4f}s, Avg time per epoch = {avg_time_0:.4f}s")

    #=====Run training on GPU 1 with AMP=True=====
    time_taken_1, avg_time_1 = train_yolo(
        device_id=[0,1],
        amp=True,
        model_name=model_name,
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz
    )
    print(f"GPU 1 (AMP=True): Total time = {time_taken_1:.4f}s, Avg time per epoch = {avg_time_1:.4f}s")

    # =====Calculating speedup and efficiency=====
    speedup = time_taken_0 / time_taken_1 if time_taken_1 > 0 else float('inf')
    efficiency = (1 / speedup) * 100  

    print(f"\nPerformance Metrics:")
    print(f"Speedup (AMP=True vs AMP=False): {speedup:.4f}x")
    print(f"Efficiency (AMP=True relative to AMP=False): {efficiency:.2f}%")
    print(f"Time Taken (GPU 0, AMP=False): {time_taken_0:.4f}s")
    print(f"Time Taken (GPU 1, AMP=True): {time_taken_1:.4f}s")

if __name__ == "__main__":
    if torch.cuda.device_count() < 2:
        print("Error: At least 2 GPUs are required.")
        exit(1)
    main()