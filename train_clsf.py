from ultralytics import YOLO


model = YOLO("yolo11n-cls.pt")
results = model.train(data="/home/moo/PycharmProjects/clsf_yolo/105_classes_pins_dataset", epochs=100, imgsz=256)
