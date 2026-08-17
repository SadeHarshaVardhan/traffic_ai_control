from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model("road1 image 1.jpg", conf=0.25)

for result in results:
    result.show()