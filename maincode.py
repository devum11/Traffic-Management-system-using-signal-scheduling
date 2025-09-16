import cv2
import numpy as np
import torch
from ultralytics import YOLO
import random
import os
import time

# Load pre-trained vehicle detection model
vehicle_model = YOLO("vehicle_counter.pt")

# Function to process video
def process_traffic_video(video_path, output_folder="output_frames", target_fps=30):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video loaded: {frame_width}x{frame_height} at {original_fps} FPS, {total_frames} frames")
    print(f"Normalizing output video to {target_fps} FPS")
    
    # Define virtual red line position (adjust y-coordinate based on video)
    red_line_y = int(frame_height * 0.7)  # Position at 70% of frame height
    
    # Initialize threshold line position to be updated based on detection
    threshold_line_y = frame_height  # Start at bottom, will be updated based on detection
    
    # Distance calibration factors (adjust based on your scene)
    # We'll use a simple linear model: distance (m) = scale_factor * (frame_height - y_position)
    scale_factor = 0.2  # Adjust based on actual measurements
    
    # Setup for video output with normalized FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_folder, "processed_traffic_normalized.mp4")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_width, frame_height))
    
    # For tracking violations
    violators = []  # List to store vehicles that crossed red line
    
    # Calculate frame sampling rate to normalize processing
    # Process every N frames to maintain consistent real-time representation
    frame_interval = max(1, int(original_fps / target_fps))
    
    frame_count = 0
    processed_count = 0
    
    # For smoothing the threshold line position (to avoid flickering)
    threshold_positions = []
    smoothing_window = 10
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Process frames at regular intervals to normalize speed
        if frame_count % frame_interval != 0:
            continue
            
        processed_count += 1
        print(f"Processing frame {frame_count}/{total_frames} ({processed_count} processed)")
        
        # Make a copy of the frame for visualization
        display_frame = frame.copy()
        
        # For tracking the furthest detection in this frame
        furthest_detection_y = frame_height
        
        # Run vehicle detection model
        vehicle_results = vehicle_model(frame)
        
        # Draw the virtual red line
        cv2.line(display_frame, (0, red_line_y), (frame_width, red_line_y), (0, 0, 255), 3)  # Red line
        
        # Draw bounding boxes for detected vehicles and check violations
        for result in vehicle_results:
            for box in result.boxes.xyxy.tolist():  # Convert tensor to list
                x1, y1, x2, y2 = map(int, box)  # Convert float to int
                vehicle_bottom_y = y2  # Bottom y-coordinate
                
                # Update furthest detection (smallest y-value = furthest from camera)
                if y1 < furthest_detection_y:
                    furthest_detection_y = y1
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate approximate distance based on y-position
                distance = round(scale_factor * (frame_height - y1), 1)
                cv2.putText(display_frame, f"{distance}m", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Check if vehicle crossed the red line
                if vehicle_bottom_y > red_line_y:
                    violators.append(((x1, y1, x2, y2), frame_count))  # Store violator details with frame number
                    cv2.putText(display_frame, "Fined!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Mark violation in red
        
        # Update the threshold line position with smoothing
        if furthest_detection_y < frame_height:
            threshold_positions.append(furthest_detection_y)
            if len(threshold_positions) > smoothing_window:
                threshold_positions.pop(0)
            threshold_line_y = sum(threshold_positions) // len(threshold_positions)
        
        # Draw the threshold detection line (yellow dashed line)
        for x in range(0, frame_width, 20):
            cv2.line(display_frame, (x, threshold_line_y), (x + 10, threshold_line_y), (255, 255, 0), 2)
        
        # Calculate and display the threshold distance
        threshold_distance = round(scale_factor * (frame_height - threshold_line_y), 1)
        cv2.putText(display_frame, f"Detection Threshold: {threshold_distance}m", 
                   (10, threshold_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add timestamp (normalized to maintain consistent speed)
        normalized_time = (frame_count / original_fps)
        minutes = int(normalized_time // 60)
        seconds = int(normalized_time % 60)
        cv2.putText(display_frame, f"Time: {minutes:02d}:{seconds:02d}", 
                   (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write the frame to output video
        out.write(display_frame)
        
        # Save key frames as images
        if processed_count % 10 == 0:  # Save every 10th processed frame
            frame_path = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, display_frame)
            
        # Display the processed frame (optional - comment out for faster processing)
        cv2.imshow('Traffic Analysis', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Return the violators for analysis
    return violators, output_path, target_fps

# Main function
def main():
    # Video path - update this to your video file path
    video_path = r"C:\STudyyyyyy\SCPS\FinalYearProject\License-Plate recognition\Signal Scheduling algo\traffic.mp4"  # Replace with your video path
    
    # Set the target FPS for normalized output
    target_fps = 30  # Standard video frame rate
    
    print(f"Processing video: {video_path}")
    start_time = time.time()
    
    # Process the video with normalized speed
    violators, output_path, fps = process_traffic_video(
        video_path, 
        target_fps=target_fps
    )
    
    processing_time = time.time() - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Processed video saved to: {output_path} at {fps} FPS")
    
    # Print violation details
    print("\n🚨 Vehicles Crossing the Red Line (Fined):")
    for bbox, frame in violators:
        print(f"Frame: {frame} | Bounding Box: {bbox}")

if __name__ == "__main__":
    main()