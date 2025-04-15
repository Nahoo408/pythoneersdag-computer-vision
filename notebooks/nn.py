from ultralytics import YOLO
from pathlib import Path
import cv2


data_path = Path("../data/images/3_neural_networks/")

paths = [

    str(data_path / "image_01.jpg"),
    str(data_path / "image_02.jpg"),
    str(data_path / "image_03.jpg"),
    str(data_path / "image_04.jpg"),
]

model = YOLO("yolo11n.pt")  # load an official model

results = model.train(data="lvis.yaml", epochs=5, imgsz=640)

for path in paths:
    results = model(path)  # predict on an image

    for result in results:
        im = result.plot()

    cv2.imshow("image", im)
    cv2.waitKey(0)
