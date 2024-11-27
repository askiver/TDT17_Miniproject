from ultralytics import YOLO

# Load a pre-trained YOLOv11 model (e.g., small version)
model = YOLO('yolo11s.pt')  # YOLOv11 will use similar weights; update as needed.

if __name__ == '__main__':

    # Train the model
    model.train(
        data='data/data.yaml',  # Path to your dataset YAML file
        epochs=50,         # Number of training epochs
        imgsz=1024,        # Input image size (matches your data, optional resizing handled)
        batch=8,          # Batch size
        device=0,          # GPU (0) or CPU (-1)
        cache = True,      # Cache images for faster training
        save_period = 10,  # Save model every 10 epochs
        optimizer = 'AdamW',
        seed = 0,
        pretrained = False,
        deterministic = False,
        #fraction = 1.0,
        plots = True,
        #patience = 10,
        single_cls = True,
        lr0 = 0.0001,
        lrf = 0.05,
        freeze = 0,
        close_mosaic = 15,
        flipud = 0.0,
        hsv_h = 0.0,
        hsv_s = 0.0,
        hsv_v = 0.0,
        translate = 0.0,
        scale = 0.0
    )