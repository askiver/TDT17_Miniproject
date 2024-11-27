from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train7/weights/best.pt')

    metrics = model.val(split='test')
    print(metrics)