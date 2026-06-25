from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt') 
    model.train(
        data='C:/fyp basic models/durian.yaml', 
        epochs=150,
        imgsz=640,
        batch=-1,
        name='C:/fyp basic models/runs/detect/durian/v8',
        patience=20,  
        device=0      
    )

    metrics = model.val()
    print(f"训练完成,MAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    train_model()