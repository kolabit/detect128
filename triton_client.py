from ultralytics import YOLO


def main():

    model = YOLO("http://localhost:8000/detect128", task="detect")
    results = model("data/dataset/images/test")
    print(results)


if __name__ == "__main__":
    main()
