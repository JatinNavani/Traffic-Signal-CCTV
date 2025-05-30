import cv2
import numpy as np
from ultralytics import YOLO
import urllib.request
import os

class TrafficZoneYOLOManager:
    def __init__(self):
        self.video_path = None
        self.cap = None
        self.frame = None
        self.zones = []  # List to store zone rectangles
        self.current_zone = None
        self.drawing = False
        self.editing_zone = None
        self.edit_mode = False
        self.start_point = (0, 0)
        self.resize_handle = None  # Which handle is being dragged
        self.moving_zone = False   # Whether zone is being moved
        self.move_offset = (0, 0)  # Offset for moving
        
        # Colors for zones
        self.zone_color = (0, 255, 0)  # Green
        self.selected_color = (0, 0, 255)  # Red
        self.text_color = (255, 255, 255)  # White
        self.handle_color = (255, 0, 255)  # Magenta for handles
        self.detection_color = (255, 0, 0)  # Blue for detections
        
        # YOLO model
        self.model = None
        self.detection_enabled = False
        
        # Vehicle classes from COCO dataset (cars, trucks, buses, motorcycles)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Zone statistics
        self.zone_counts = {}  # Store vehicle counts per zone
        self.detection_history = []  # Store detection history
        
    def load_yolo_model(self, model_size='n'):
        """Load YOLOv8 model (automatically downloads if not present)"""
        try:
            model_name = f'yolov8{model_size}.pt'
            print(f"Loading YOLOv8 model: {model_name}")
            print("Note: Model will be automatically downloaded on first use...")
            
            self.model = YOLO(model_name)
            print(f"✓ YOLOv8{model_size} model loaded successfully!")
            
            # Initialize zone counts
            for i in range(len(self.zones)):
                self.zone_counts[i] = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
                
            return True
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Make sure you have ultralytics installed: pip install ultralytics")
            return False
    
    def load_video(self, video_path):
        """Load video file"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False
            
        # Read first frame
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return False
            
        print(f"Video loaded successfully: {video_path}")
        return True
    
    def point_in_zone(self, point, zone):
        """Check if a point is inside a zone"""
        x, y = point
        x1, y1 = zone[0]
        x2, y2 = zone[1]
        
        # Normalize coordinates
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        return min_x <= x <= max_x and min_y <= y <= max_y
    
    def detect_vehicles_in_zones(self, frame):
        """Perform vehicle detection only within defined zones"""
        if not self.model or not self.detection_enabled or not self.zones:
            return frame, {}
        
        # Reset zone counts
        zone_detections = {}
        for i in range(len(self.zones)):
            zone_detections[i] = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0, 'detections': []}
        
        try:
            # Run YOLO inference
            results = self.model(frame, verbose=False)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Only process vehicle classes with good confidence
                        if class_id in self.vehicle_classes and confidence > 0.5:
                            vehicle_type = self.vehicle_classes[class_id]
                            
                            # Calculate center point of detection
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Check which zones contain this detection
                            for zone_idx, zone in enumerate(self.zones):
                                if self.point_in_zone((center_x, center_y), zone):
                                    # Count detection in this zone
                                    zone_detections[zone_idx][vehicle_type] += 1
                                    zone_detections[zone_idx]['detections'].append({
                                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                        'confidence': confidence,
                                        'class': vehicle_type,
                                        'center': (center_x, center_y)
                                    })
            
            # Draw detections only for vehicles in zones
            annotated_frame = self.draw_zone_detections(frame, zone_detections)
            
            # Update zone counts
            self.zone_counts = zone_detections
            
            return annotated_frame, zone_detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, {}
    
    def draw_zone_detections(self, frame, zone_detections):
        """Draw detection boxes and labels for vehicles in zones"""
        annotated_frame = frame.copy()
        
        for zone_idx, detections in zone_detections.items():
            for detection in detections['detections']:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                vehicle_class = detection['class']
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.detection_color, 2)
                
                # Draw label with confidence
                label = f"{vehicle_class}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), self.detection_color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw center point
                center_x, center_y = detection['center']
                cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 255, 255), -1)
        
        return annotated_frame
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zone creation and editing"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # First check if we're on a resize handle of selected zone
            if self.editing_zone is not None:
                handle = self.get_resize_handle(x, y, self.editing_zone)
                if handle:
                    self.resize_handle = handle
                    print(f"Started resizing zone {self.editing_zone} with handle {handle}")
                    return
            
            # Check if clicking on existing zone for editing/moving
            clicked_zone = self.get_zone_at_point(x, y)
            
            if clicked_zone is not None:
                if self.editing_zone == clicked_zone:
                    # Start moving the already selected zone
                    self.moving_zone = True
                    zone = self.zones[clicked_zone]
                    zone_center_x = (zone[0][0] + zone[1][0]) // 2
                    zone_center_y = (zone[0][1] + zone[1][1]) // 2
                    self.move_offset = (x - zone_center_x, y - zone_center_y)
                    print(f"Started moving zone {clicked_zone}")
                else:
                    # Select different zone
                    self.editing_zone = clicked_zone
                    self.edit_mode = True
                    print(f"Selected zone {clicked_zone} for editing")
            else:
                # Start creating new zone
                self.editing_zone = None
                self.edit_mode = False
                self.drawing = True
                self.start_point = (x, y)
                self.current_zone = [(x, y), (x, y)]
                print(f"Started creating zone at ({x}, {y})")
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.resize_handle and self.editing_zone is not None:
                # Resize selected zone
                self.resize_zone(self.editing_zone, self.resize_handle, x, y)
            elif self.moving_zone and self.editing_zone is not None:
                # Move selected zone
                zone = self.zones[self.editing_zone]
                width = abs(zone[1][0] - zone[0][0])
                height = abs(zone[1][1] - zone[0][1])
                
                # Calculate new position
                new_center_x = x - self.move_offset[0]
                new_center_y = y - self.move_offset[1]
                
                # Update zone position
                self.zones[self.editing_zone] = [
                    (new_center_x - width//2, new_center_y - height//2),
                    (new_center_x + width//2, new_center_y + height//2)
                ]
            elif self.drawing and self.current_zone:
                # Update current zone while drawing
                self.current_zone[1] = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.resize_handle:
                # Finish resizing
                print(f"Finished resizing zone {self.editing_zone}")
                self.resize_handle = None
            elif self.moving_zone:
                # Finish moving
                print(f"Finished moving zone {self.editing_zone}")
                self.moving_zone = False
                self.move_offset = (0, 0)
            elif self.drawing:
                # Finish creating zone
                self.current_zone[1] = (x, y)
                
                # Only add zone if it has reasonable size
                if abs(x - self.start_point[0]) > 10 and abs(y - self.start_point[1]) > 10:
                    self.zones.append(self.current_zone)
                    # Initialize zone count for new zone
                    zone_idx = len(self.zones) - 1
                    self.zone_counts[zone_idx] = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
                    print(f"Zone created: {self.current_zone}")
                
                self.drawing = False
                self.current_zone = None
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to delete zone
            clicked_zone = self.get_zone_at_point(x, y)
            if clicked_zone is not None:
                del self.zones[clicked_zone]
                # Remove zone count
                if clicked_zone in self.zone_counts:
                    del self.zone_counts[clicked_zone]
                # Reindex remaining zone counts
                new_counts = {}
                for old_idx, counts in self.zone_counts.items():
                    if old_idx > clicked_zone:
                        new_counts[old_idx - 1] = counts
                    elif old_idx < clicked_zone:
                        new_counts[old_idx] = counts
                self.zone_counts = new_counts
                
                print(f"Deleted zone {clicked_zone}")
                self.editing_zone = None
                self.edit_mode = False
                self.resize_handle = None
                self.moving_zone = False
    
    def get_resize_handle(self, x, y, zone_idx):
        """Check if click is on a resize handle and return handle type"""
        if zone_idx >= len(self.zones):
            return None
            
        zone = self.zones[zone_idx]
        x1, y1 = zone[0]
        x2, y2 = zone[1]
        
        # Normalize coordinates
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        handle_size = 8
        
        # Check corner handles
        corners = {
            'top_left': (min_x, min_y),
            'top_right': (max_x, min_y),
            'bottom_left': (min_x, max_y),
            'bottom_right': (max_x, max_y)
        }
        
        for handle_name, (hx, hy) in corners.items():
            if abs(x - hx) <= handle_size and abs(y - hy) <= handle_size:
                return handle_name
        
        # Check edge handles
        mid_x = (min_x + max_x) // 2
        mid_y = (min_y + max_y) // 2
        
        edges = {
            'top': (mid_x, min_y),
            'bottom': (mid_x, max_y),
            'left': (min_x, mid_y),
            'right': (max_x, mid_y)
        }
        
        for handle_name, (hx, hy) in edges.items():
            if abs(x - hx) <= handle_size and abs(y - hy) <= handle_size:
                return handle_name
        
        return None

    def get_zone_at_point(self, x, y):
        """Check if point is inside any zone and return zone index"""
        for i, zone in enumerate(self.zones):
            x1, y1 = zone[0]
            x2, y2 = zone[1]
            
            # Normalize coordinates
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return i
        return None
    
    def resize_zone(self, zone_idx, handle, new_x, new_y):
        """Resize zone based on handle being dragged"""
        if zone_idx >= len(self.zones):
            return
            
        zone = self.zones[zone_idx]
        x1, y1 = zone[0]
        x2, y2 = zone[1]
        
        if handle == 'top_left':
            self.zones[zone_idx] = [(new_x, new_y), (x2, y2)]
        elif handle == 'top_right':
            self.zones[zone_idx] = [(x1, new_y), (new_x, y2)]
        elif handle == 'bottom_left':
            self.zones[zone_idx] = [(new_x, y1), (x2, new_y)]
        elif handle == 'bottom_right':
            self.zones[zone_idx] = [(x1, y1), (new_x, new_y)]
        elif handle == 'top':
            self.zones[zone_idx] = [(x1, new_y), (x2, y2)]
        elif handle == 'bottom':
            self.zones[zone_idx] = [(x1, y1), (x2, new_y)]
        elif handle == 'left':
            self.zones[zone_idx] = [(new_x, y1), (x2, y2)]
        elif handle == 'right':
            self.zones[zone_idx] = [(x1, y1), (new_x, y2)]
    
    def draw_zones(self, frame):
        """Draw all zones on the frame"""
        display_frame = frame.copy()
        
        # Draw existing zones
        for i, zone in enumerate(self.zones):
            x1, y1 = zone[0]
            x2, y2 = zone[1]
            
            # Choose color based on selection
            color = self.selected_color if i == self.editing_zone else self.zone_color
            
            # Draw rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw zone label with vehicle counts
            if i in self.zone_counts:
                counts = self.zone_counts[i]
                total_vehicles = sum([counts.get(vtype, 0) for vtype in ['car', 'truck', 'bus', 'motorcycle']])
                label = f"Zone {i+1} ({total_vehicles} vehicles)"
            else:
                label = f"Zone {i+1}"
                
            label_pos = (min(x1, x2), min(y1, y2) - 10)
            cv2.putText(display_frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.text_color, 2)
            
            # Draw resize handles for selected zone
            if i == self.editing_zone:
                self.draw_resize_handles(display_frame, zone)
        
        # Draw current zone being created
        if self.drawing and self.current_zone:
            x1, y1 = self.current_zone[0]
            x2, y2 = self.current_zone[1]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), self.zone_color, 2)
        
        return display_frame
    
    def draw_resize_handles(self, frame, zone):
        """Draw resize handles on selected zone"""
        x1, y1 = zone[0]
        x2, y2 = zone[1]
        
        # Normalize coordinates
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        handle_size = 6
        
        # Corner handles
        corners = [
            (min_x, min_y),  # top_left
            (max_x, min_y),  # top_right
            (min_x, max_y),  # bottom_left
            (max_x, max_y)   # bottom_right
        ]
        
        # Edge handles
        mid_x = (min_x + max_x) // 2
        mid_y = (min_y + max_y) // 2
        edges = [
            (mid_x, min_y),  # top
            (mid_x, max_y),  # bottom
            (min_x, mid_y),  # left
            (max_x, mid_y)   # right
        ]
        
        # Draw all handles
        all_handles = corners + edges
        for hx, hy in all_handles:
            cv2.rectangle(frame, 
                         (hx - handle_size, hy - handle_size),
                         (hx + handle_size, hy + handle_size),
                         self.handle_color, -1)
            cv2.rectangle(frame, 
                         (hx - handle_size, hy - handle_size),
                         (hx + handle_size, hy + handle_size),
                         (0, 0, 0), 1)
    
    def add_instructions(self, frame):
        """Add instruction text to frame"""
        instructions = [
            "Instructions:",
            "- Click and drag to create zone",
            "- Left click zone to select",
            "- Drag handles to resize zone",
            "- Drag zone center to move",
            "- Right click zone to delete",
            "- Press 'd' to toggle detection",
            "- Press 'r' to reset all zones",
            "- Press 'q' to quit"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Show zone count and detection status
        zone_info = f"Total Zones: {len(self.zones)}"
        detection_status = f"Detection: {'ON' if self.detection_enabled else 'OFF'}"
        model_status = f"Model: {'Loaded' if self.model else 'Not Loaded'}"
        
        cv2.putText(frame, zone_info, (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, detection_status, (10, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.detection_enabled else (0, 0, 255), 2)
        cv2.putText(frame, model_status, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.model else (0, 0, 255), 2)
        
        return frame
    
    def add_zone_statistics(self, frame):
        """Add zone-wise vehicle statistics to frame"""
        if not self.zone_counts:
            return frame
        
        # Create statistics panel
        stats_x = frame.shape[1] - 300
        stats_y = 30
        
        cv2.putText(frame, "Zone Statistics:", (stats_x, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = stats_y + 30
        for zone_idx, counts in self.zone_counts.items():
            if zone_idx < len(self.zones):  # Make sure zone still exists
                zone_text = f"Zone {zone_idx + 1}:"
                cv2.putText(frame, zone_text, (stats_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 20
                
                for vehicle_type, count in counts.items():
                    if vehicle_type != 'detections' and count > 0:
                        count_text = f"  {vehicle_type}: {count}"
                        cv2.putText(frame, count_text, (stats_x, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        y_offset += 15
                
                y_offset += 10  # Extra space between zones
        
        return frame
    
    def run(self):
        """Main loop to run the zone manager"""
        if not self.cap or not self.cap.isOpened():
            print("No video loaded. Please load a video first.")
            return
        
        # Create window and set mouse callback
        cv2.namedWindow('Traffic Zone Manager with YOLO', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Traffic Zone Manager with YOLO', self.mouse_callback)
        
        print("\n=== Traffic Zone Manager with YOLOv8 Started ===")
        print("Instructions:")
        print("- Click and drag to create rectangular zones")
        print("- Left click on a zone to select it")
        print("- Drag the corner/edge handles to resize selected zone")
        print("- Click and drag zone center to move it")
        print("- Right click on a zone to delete it")
        print("- Press 'd' to toggle vehicle detection")
        print("- Press 'l' to load YOLO model")
        print("- Press 'r' to reset all zones")
        print("- Press 'q' to quit")
        print("- Press SPACE to pause/unpause video")
        
        paused = True  # Start paused to allow zone creation
        
        while True:
            if not paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, self.frame = self.cap.read()
            
            if self.frame is not None:
                # Perform vehicle detection in zones
                if self.detection_enabled and self.model:
                    display_frame, zone_detections = self.detect_vehicles_in_zones(self.frame)
                else:
                    display_frame = self.frame.copy()
                
                # Draw zones on frame
                display_frame = self.draw_zones(display_frame)
                
                # Add instructions
                display_frame = self.add_instructions(display_frame)
                
                # Add zone statistics
                display_frame = self.add_zone_statistics(display_frame)
                
                # Show frame
                cv2.imshow('Traffic Zone Manager with YOLO', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset all zones
                self.zones = []
                self.zone_counts = {}
                self.editing_zone = None
                self.edit_mode = False
                print("All zones reset")
            elif key == ord(' '):
                # Toggle pause
                paused = not paused
                print(f"Video {'paused' if paused else 'playing'}")
            elif key == ord('d'):
                # Toggle detection
                if self.model:
                    self.detection_enabled = not self.detection_enabled
                    print(f"Detection {'enabled' if self.detection_enabled else 'disabled'}")
                else:
                    print("Please load YOLO model first (press 'l')")
            elif key == ord('l'):
                # Load YOLO model
                print("Loading YOLOv8 model...")
                if self.load_yolo_model('n'):  # Load nano model (fastest)
                    print("YOLO model loaded! Press 'd' to enable detection.")
                else:
                    print("Failed to load YOLO model")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.zones:
            print(f"\nFinal zones created: {len(self.zones)}")
            for i, zone in enumerate(self.zones):
                print(f"Zone {i+1}: {zone}")
                if i in self.zone_counts:
                    counts = self.zone_counts[i]
                    total = sum([counts.get(vtype, 0) for vtype in ['car', 'truck', 'bus', 'motorcycle']])
                    print(f"  Vehicles detected: {total}")
                    for vtype, count in counts.items():
                        if vtype != 'detections' and count > 0:
                            print(f"    {vtype}: {count}")
        else:
            print("\nNo zones created")


# Example usage
if __name__ == "__main__":
    # Create zone manager
    zone_manager = TrafficZoneYOLOManager()
    
    # Load video (replace with your video path)
    video_path = "D:/Traffic/traffix/test.mp4"  # Change this to your video file path
    
    if zone_manager.load_video(video_path):
        zone_manager.run()
    else:
        print("Please update the video_path variable with a valid video file path")
        print("Example: video_path = '/path/to/your/video.mp4'")
        print("\nRequired packages:")
        print("pip install ultralytics opencv-python")