import cv2
import os
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import requests
import base64
from io import BytesIO
from PIL import Image
from supabase import create_client, Client

# Try to load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

# ---------------------------
# 1️⃣ Configuration
# ---------------------------
# Hugging Face API Configuration
HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/your-model-name")
HF_API_KEY = os.getenv("HF_API_KEY", "")  # Required: Your Hugging Face API key

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://kwjxcodhmxlsvkftsgcg.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt3anhjb2RobXhsc3ZrZnRzZ2NnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjMxMzE1NzQsImV4cCI6MjA3ODcwNzU3NH0.bT7aPEpxwMS9UFbLetj_7sA-Jd520todnozEVaCTmys")

# Camera Configuration
CAMERA_INDEX = 0  # Laptop camera (change to 1, 2, etc. if needed)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Detection Configuration
PERSON_DETECTION_INTERVAL = 0.1  # Check for person every 0.1 seconds
PROCESS_INTERVAL = 2  # Process detected person every 2 seconds (to avoid too many API calls)

# ---------------------------
# 2️⃣ Initialize Supabase
# ---------------------------
supabase: Client = None
if SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Connected to Supabase")
    except Exception as e:
        print(f"⚠ Supabase connection error: {str(e)}")
        print("⚠ Continuing without Supabase logging...")
else:
    print("⚠ SUPABASE_KEY not set. Set it as environment variable to enable database logging.")

# ---------------------------
# 3️⃣ Initialize Person Detection Model (YOLO)
# ---------------------------
# Load YOLO model for person detection
# Using YOLOv8n for person detection (lightweight and fast)
person_detection_model = YOLO("yolov8n.pt")  # This will download automatically if not present
print("✓ Loaded YOLO person detection model")

# ---------------------------
# 4️⃣ Initialize Laptop Camera
# ---------------------------
def initialize_camera():
    """Initialize laptop camera"""
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        time.sleep(0.5)  # Give camera time to initialize
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open camera at index {CAMERA_INDEX}")
            print("  Try changing CAMERA_INDEX to 1, 2, etc. if you have multiple cameras")
            return None
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
        
        # Get actual camera properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✓ Camera opened successfully")
        print(f"  Resolution: {actual_width}x{actual_height}")
        print(f"  FPS: {actual_fps}")
        
        # Warm-up: Read a few frames
        print("  Warming up camera...")
        for i in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✓ Frame read successful")
                break
            time.sleep(0.2)
        
        return cap
        
    except Exception as e:
        print(f"✗ Error initializing camera: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Initialize camera
print(f"\n{'='*50}")
print(f"Initializing laptop camera...")
print(f"  Camera index: {CAMERA_INDEX}")
print(f"  Target resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
print(f"  Hugging Face API: {HF_API_URL}")
print(f"{'='*50}\n")

cap = initialize_camera()

if cap is None:
    print("\nTroubleshooting:")
    print("  1. Make sure your laptop camera is connected and working")
    print("  2. Check if another application is using the camera")
    print("  3. Try changing CAMERA_INDEX in the code (0, 1, 2, etc.)")
    exit(1)

# ---------------------------
# 5️⃣ Send Image to Hugging Face API
# ---------------------------
def send_image_to_huggingface(image_path):
    """
    Send image to Hugging Face API for person identification and PPE detection.
    Returns: dict with person_id, name, ppe_status, confidence, or None if failed
    """
    if not HF_API_KEY:
        print("  ✗ HF_API_KEY not set. Cannot send to Hugging Face.")
        return None
    
    try:
        # Read image file
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Prepare headers with API key
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/octet-stream"
        }
        
        # Send POST request to Hugging Face Inference API
        print("  → Sending image to Hugging Face API...")
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=image_bytes,
            timeout=30
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                # Parse the response from Hugging Face
                # Expected format (adjust based on your Hugging Face model output):
                # {
                #   "person_id": "EMP001",
                #   "name": "John Doe",
                #   "ppe": {
                #     "helmet": true,
                #     "vest": true,
                #     "mask": false
                #   },
                #   "confidence": 0.95,
                #   "raw_ppe_detections": ["Hardhat", "Safety Vest", "NO-Mask"]
                # }
                
                print("  ✓ Received response from Hugging Face")
                
                # Extract data from response
                person_id = result.get("person_id")
                name = result.get("name", "Unknown")
                confidence = result.get("confidence", 0.0)
                
                # Extract PPE status
                ppe_data = result.get("ppe", {})
                ppe_status = {
                    "helmet": ppe_data.get("helmet"),
                    "vest": ppe_data.get("vest"),
                    "mask": ppe_data.get("mask")
                }
                
                raw_ppe_detections = result.get("raw_ppe_detections", [])
                
                return {
                    "person_id": person_id,
                    "name": name,
                    "ppe_status": ppe_status,
                    "confidence": confidence,
                    "raw_ppe_detections": raw_ppe_detections
                }
                
            except Exception as e:
                print(f"  ✗ Error parsing response: {str(e)}")
                print(f"  Response: {response.text[:200]}")
                return None
        else:
            print(f"  ✗ API request failed with status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timeout - API took too long to respond")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request error: {str(e)}")
        return None
    except Exception as e:
        print(f"  ✗ Error sending image: {str(e)}")
        return None

# ---------------------------
# 6️⃣ Send Results to Supabase
# ---------------------------
def send_to_supabase(detection_result):
    """Send detection results to Supabase"""
    if not supabase:
        print("  ⚠ Supabase not configured - skipping database save")
        return False
    
    try:
        detection_data = {
            "person_id": detection_result.get("person_id"),
            "name": detection_result.get("name") if detection_result.get("name") != "Unknown" else None,
            "helmet": detection_result.get("ppe_status", {}).get("helmet"),
            "vest": detection_result.get("ppe_status", {}).get("vest"),
            "mask": detection_result.get("ppe_status", {}).get("mask"),
            "raw_ppe_detections": detection_result.get("raw_ppe_detections"),
            "image_url": None,  # You can upload to Supabase Storage and set URL here
            "confidence": float(detection_result.get("confidence", 0.0)) if detection_result.get("confidence") else None,
            "timestamp": datetime.now().isoformat()
        }
        
        response = supabase.table("detections_log").insert(detection_data).execute()
        
        if response.data:
            print(f"  ✓ Data saved to Supabase successfully!")
            print(f"    Record ID: {response.data[0]['id']}")
            return True
        else:
            print(f"  ⚠ Warning: Data sent but no response received")
            return False
            
    except Exception as e:
        print(f"  ✗ Supabase error: {str(e)}")
        return False

# ---------------------------
# 7️⃣ Main Detection Loop
# ---------------------------
print(f"\n{'='*50}")
print(f"📹 Starting person detection and monitoring...")
print(f"  Press 'q' to quit")
print(f"{'='*50}\n")

image_count = 0
last_person_detection_time = 0
last_process_time = 0
person_detected = False

try:
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("⚠ Warning: Could not read frame from camera")
            time.sleep(0.5)
            continue
        
        current_time = time.time()
        
        # Check for person detection at regular intervals
        if current_time - last_person_detection_time >= PERSON_DETECTION_INTERVAL:
            last_person_detection_time = current_time
            
            # Run YOLO person detection
            results = person_detection_model(frame, verbose=False, imgsz=640)
            
            # Check if person is detected
            person_detected = False
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        
                        if class_name.lower() == 'person':
                            person_detected = True
                            break
                    if person_detected:
                        break
        
        # If person detected, process image and send to Hugging Face
        if person_detected:
            # Process at intervals to avoid too many API calls
            if current_time - last_process_time >= PROCESS_INTERVAL:
                last_process_time = current_time
                
                # Save temporary image
                temp_image_path = "temp_detection.jpg"
                cv2.imwrite(temp_image_path, frame)
                
                image_count += 1
                
                print(f"\n{'='*50}")
                print(f"📸 Person detected! Processing image #{image_count}...")
                
                # Send image to Hugging Face API
                hf_result = send_image_to_huggingface(temp_image_path)
                
                if hf_result:
                    print(f"\n📊 Detection Results:")
                    print(f"   Person: {hf_result.get('name', 'Unknown')}")
                    print(f"   ID: {hf_result.get('person_id', 'N/A')}")
                    print(f"   Confidence: {hf_result.get('confidence', 0.0):.2%}")
                    print(f"   PPE Status:")
                    ppe_status = hf_result.get('ppe_status', {})
                    print(f"     Helmet: {ppe_status.get('helmet')}")
                    print(f"     Vest: {ppe_status.get('vest')}")
                    print(f"     Mask: {ppe_status.get('mask')}")
                    print(f"   Raw detections: {hf_result.get('raw_ppe_detections', [])}")
                    
                    # Send results to Supabase
                    print(f"\n💾 Sending results to Supabase...")
                    send_to_supabase(hf_result)
                    
                    print(f"✅ Image #{image_count} processed successfully!")
                else:
                    print(f"⚠ Image #{image_count} captured but Hugging Face API request failed")
                
                # Clean up temporary image file
                try:
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                except:
                    pass
                
                print(f"  Next detection in {PROCESS_INTERVAL} seconds...")
                print(f"{'='*50}\n")
            
            # Draw bounding boxes on frame for visualization
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, "Person Detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Person Detection & PPE Monitoring', annotated_frame)
        else:
            # Show frame without annotations if no person detected
            cv2.putText(frame, "No person detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Person Detection & PPE Monitoring', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⚠ Keyboard interrupt received. Stopping camera...")

except Exception as e:
    print(f"\n✗ Unexpected error: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    # Release camera and close windows
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"\n✓ Camera closed. Total images processed: {image_count}")
