import streamlit as st
import cv2
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import pandas as pd


conn = sqlite3.connect("traffic.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS traffic_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    cars INTEGER,
    bikes INTEGER,
    bus INTEGER,
    truck INTEGER,
    no_helmet INTEGER
)
""")

def insert_data(data):
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO traffic_data (date, cars, bikes, bus, truck, no_helmet)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (date, data["cars"], data["bikes"], data["bus"], data["truck"], data["no_helmet"]))
    conn.commit()

def get_data():
    cursor.execute("SELECT * FROM traffic_data ORDER BY id ASC")
    return cursor.fetchall()


vehicle_model = YOLO("models/yolo26n.pt")
helmet_model = YOLO("models/best1.pt")

names = vehicle_model.names

CAR, BIKE, BUS, TRUCK = 2, 3, 5, 7
vehicle_classes = [CAR, BIKE, BUS, TRUCK]


st.title("🚦 Smart Traffic System")

video = st.file_uploader("Upload Video")

col1, col2 = st.columns(2)
start = col1.button("▶️ Start")
stop = col2.button("⏹ Stop")


if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True

if stop:
    st.session_state.run = False


if video and st.session_state.run:

    with open("temp.mp4", "wb") as f:
        f.write(video.read())

    cap = cv2.VideoCapture("temp.mp4")

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    counted_ids = set()
    no_helmet_ids = set()

    car = bike = bus = truck = no_helmet = 0
    line_y = 300
    frame_id = 0

    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        results = vehicle_model.track(frame, persist=True)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None

            if conf < 0.5 or track_id is None:
                continue

            if cls not in vehicle_classes:
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            center_y = (y1+y2)//2

          
            if center_y > line_y and track_id not in counted_ids:
                if cls == CAR: car += 1
                elif cls == BIKE: bike += 1
                elif cls == BUS: bus += 1
                elif cls == TRUCK: truck += 1
                counted_ids.add(track_id)

            
            if cls == BIKE:
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    h_results = helmet_model(crop)[0]
                    for hbox in h_results.boxes:
                        hcls = int(hbox.cls[0])
                        if hcls == 1 and track_id not in no_helmet_ids:
                            no_helmet += 1
                            no_helmet_ids.add(track_id)

            
            label = f"{names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame, label, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

        
        cv2.line(frame, (0,line_y),(frame.shape[1],line_y),(0,255,255),2)

        
        stats = {
            "cars": car,
            "bikes": bike,
            "bus": bus,
            "truck": truck,
            "no_helmet": no_helmet
        }

        
        frame_placeholder.image(frame, channels="BGR")
        stats_placeholder.json(stats) 

        
        if frame_id % 60 == 0:
            insert_data(stats)

    cap.release()

    insert_data(stats)


st.subheader("📊 History Data")

history = get_data()

if history:
    df = pd.DataFrame(history, columns=[
        "id","date","cars","bikes","bus","truck","no_helmet"
    ])

    st.dataframe(df.tail(20))

else:
    st.info("No data yet. Run a video to generate stats.")
