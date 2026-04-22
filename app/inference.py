import cv2
from ultralytics import YOLO

class TrafficSystem:
    def __init__(self):
        self.vehicle_model = YOLO("models/yolo26n.pt")
        self.helmet_model = YOLO("models/best1.pt")

        self.names = self.vehicle_model.names

        # class IDs
        self.CAR = 2
        self.BIKE = 3
        self.BUS = 5
        self.TRUCK = 7

        self.vehicle_classes = [2,3,5,7]

        # counters
        self.counted_ids = set()
        self.no_helmet_ids = set()

        self.car = 0
        self.bike = 0
        self.bus = 0
        self.truck = 0
        self.no_helmet = 0

        self.line_y = 300

    def process_frame(self, frame):
        results = self.vehicle_model.track(frame, persist=True)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None

            if conf < 0.5 or track_id is None:
                continue

            if cls not in self.vehicle_classes:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            center_y = (y1+y2)//2

            # count
            if center_y > self.line_y and track_id not in self.counted_ids:
                if cls == self.CAR: self.car += 1
                elif cls == self.BIKE: self.bike += 1
                elif cls == self.BUS: self.bus += 1
                elif cls == self.TRUCK: self.truck += 1

                self.counted_ids.add(track_id)

            # helmet check
            if cls == self.BIKE:
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    h_results = self.helmet_model(crop)[0]

                    for hbox in h_results.boxes:
                        hcls = int(hbox.cls[0])
                        if hcls == 1 and track_id not in self.no_helmet_ids:
                            self.no_helmet += 1
                            self.no_helmet_ids.add(track_id)

        return {
            "cars": self.car,
            "bikes": self.bike,
            "bus": self.bus,
            "truck": self.truck,
            "no_helmet": self.no_helmet
        }
