import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random

# Load pre-trained vehicle detection model
vehicle_model = YOLO("vehicle_counter.pt")

# Define lane images
lane_images = {
    "Lane 1": r"C:\STudyyyyyy\SCPS\FinalYearProject\License-Plate recognition\Signal Scheduling algo\vehicle_counter_test\images\yt-iJZcjZD0fw0-0616_jpg.rf.c2de9664a543d1e1003039c5c995f71d.jpg",
    "Lane 2": r"C:\STudyyyyyy\SCPS\FinalYearProject\License-Plate recognition\Signal Scheduling algo\vehicle_counter_test\images\yt-iJZcjZD0fw0-0776_jpg.rf.5e52284408c37a805524b00d5514a4c7.jpg",
    "Lane 3": r"C:\STudyyyyyy\SCPS\FinalYearProject\License-Plate recognition\Signal Scheduling algo\vehicle_counter_test\images\yt-iJZcjZD0fw0-0713_jpg.rf.0adc4b5fc7b5b2ac4bf8322d5825f980.jpg",
    "Lane 4": r"C:\STudyyyyyy\SCPS\FinalYearProject\License-Plate recognition\Signal Scheduling algo\vehicle_counter_test\images\yt-iJZcjZD0fw0-0221_jpg.rf.a2c181463a0e7d6d45ec99cf77b68d6a.jpg"
}

# Define virtual red line position (adjust y-coordinate based on image size)
red_line_y = 400  # Adjust as per your image resolution

# Lane dividers for dotted lines (vertical)
lane_dividers = [250, 500, 750]

# First Pass: Count vehicles in each lane
lane_vehicle_counts = {}
for lane, img_path in lane_images.items():
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image for {lane}")
        continue

    results = vehicle_model(img)
    vehicle_count = len(list(results[0].boxes.xyxy))
    lane_vehicle_counts[lane] = vehicle_count

# Assign signal timing based on vehicle density
sorted_lanes = sorted(lane_vehicle_counts.items(), key=lambda x: x[1], reverse=True)
lane_queue = [lane for lane, _ in sorted_lanes]
random.shuffle(lane_queue)

signal_timing = {}
base_time = 30  # Base green light time

for lane in lane_queue:
    count = lane_vehicle_counts[lane]
    signal_timing[lane] = base_time + (count * 2)

# Assign signal colors
total_lanes = len(sorted_lanes)
green_limit = int(total_lanes * 0.25)
yellow_limit = int(total_lanes * 0.75)

lane_colors = {}
for idx, (lane, _) in enumerate(sorted_lanes):
    if idx < green_limit:
        lane_colors[lane] = "🟢 GREEN"
    elif idx < yellow_limit:
        lane_colors[lane] = "🟡 YELLOW"
    else:
        lane_colors[lane] = "🔴 RED"

# Second Pass: Check for violations and visualize
violators = []

for lane, img_path in lane_images.items():
    img = cv2.imread(img_path)
    if img is None:
        continue

    results = vehicle_model(img)

    # Draw lane dividers (dotted white lines)
    for x in lane_dividers:
        for y in range(0, img.shape[0], 20):
            cv2.line(img, (x, y), (x, y + 10), (255, 255, 255), 2)

    # Draw the red line
    cv2.line(img, (0, red_line_y), (img.shape[1], red_line_y), (0, 0, 255), 3)

    # Draw bounding boxes and check for red signal violations
    for result in results:
        for box in result.boxes.xyxy.tolist():
            x1, y1, x2, y2 = map(int, box)
            vehicle_bottom_y = y2

            # Draw normal bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check for red signal violation
            if lane_colors.get(lane) == "🔴 RED" and vehicle_bottom_y > red_line_y:
                violators.append((lane, (x1, y1, x2, y2)))
                cv2.putText(img, "Fined!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for violator

    # Display image with annotations
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{lane} - Vehicles: {lane_vehicle_counts[lane]} | {lane_colors[lane]}")
    plt.axis("off")
    plt.show()

# Load pre-trained RL model for future use (optional)
try:
    rl_model = torch.load("pretrained_rl_model.pth", map_location=torch.device('cpu'))
except Exception as e:
    print(f"Error loading RL model: {e}")
    rl_model = None

# Final Report
print("\n🚦 Traffic Signal Timings & Colors:")
for lane, time in signal_timing.items():
    print(f"{lane}: {time} seconds | {lane_colors[lane]} | 🚗 Vehicles: {lane_vehicle_counts[lane]}")

print("\n🚨 Vehicles Crossing the Red Line (Fined):")
for lane, bbox in violators:
    print(f"Lane: {lane} | Bounding Box: {bbox}")
