from ultralytics import YOLO


model = YOLO("yolo11n.pt")
results = model.train(data="/home/moo/PycharmProjects/update_detection_yolo/train.yaml", epochs=100, imgsz=640, batch=64)
