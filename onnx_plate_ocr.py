import time
import cv2
import numpy as np
import onnxruntime as ort
import datetime
import re
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

temp = []
last_temp_update_time = datetime.datetime.now()  # Initialize the timestamp

def preprocess_frame(frame, input_size=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # Normalize & CHW
    img = np.expand_dims(img, axis=0)
    return img

def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

def postprocess(output, original_shape, conf_threshold=0.4, iou_threshold=0.5):
    output = output[0]  # (1, 5, 8400)
    output = np.squeeze(output, axis=0)  # (5, 8400)

    boxes = []
    confidences = []

    for i in range(output.shape[1]):
        conf = output[4, i]
        if conf > conf_threshold:
            xywh = output[0:4, i]
            box = xywh_to_xyxy(xywh)
            boxes.append(box)
            confidences.append(conf)

    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes)
    confidences = np.array(confidences)

    h, w = original_shape
    boxes[:, [0, 2]] *= w / 640
    boxes[:, [1, 3]] *= h / 640

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold)

    final_boxes = []
    final_confs = []
    if len(indices) > 0:
        for i in indices:
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            final_boxes.append(boxes[idx])
            final_confs.append(confidences[idx])

    return final_boxes, final_confs
def draw_boxes(frame, boxes, confidences):
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"License Plate {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def crop_plates(frame, boxes):
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Clamp coordinates to frame size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        crops.append(crop)
    return crops

def clear_temp_after_interval(interval_seconds=20):
    global temp, last_temp_update_time
    current_time = datetime.datetime.now()
    if (current_time - last_temp_update_time).total_seconds() > interval_seconds:
        temp = []
        last_temp_update_time = current_time
        #log_info("Temp list cleared after 20 seconds")
def read_license_plate(plate_img):
    result = ocr.ocr(plate_img, cls=True)
    t = ''  # Initialize the variable before processing the OCR result
    if result:
        for result1 in result:
            if result1:
                for bbox, (text, score) in result1:
                    t += text
                # Clean the text
                text = re.sub(r'[^\w\s]|_', '', t)
                t = text.upper().replace(" ", "")
                length = len(t)
                
                # Define license plate patterns
                patterns = [
                r'[a-zA-Z]{2}\d{1}[a-zA-Z]{1}\d{4}',  # Pattern 1: char, char, digit, char, digit, digit, digit, digit    = 8
                r'[a-zA-Z]{2}\d{2}[a-zA-Z]{1}\d{4}',  # Pattern 1: char, char, digit, digit, char, digit, digit, digit, digit  = 9
                r'[a-zA-Z]{2}\d{1}[a-zA-Z]{2}\d{4}',  # Pattern 1: char, char, digit, char, char, digit, digit, digit, digit  = 9
                r'[a-zA-Z]{2}\d{2}[a-zA-Z]{2}\d{4}',  # Pattern 2: char, char, digit, digit, char, char, digit, digit, digit, digit  = 10
                r'[a-zA-Z]{2}\d{1}[a-zA-Z]{3}\d{4}',  # Pattern 2: char, char, digit, char, char, char, digit, digit, digit, digit  = 10
                r'[a-zA-Z]{2}\d{2}[a-zA-Z]{3}\d{4}',  # Pattern 2: char, char, digit, digit, char, char, char, digit, digit, digit, digit  = 11
                r'[a-zA-Z]{2}\d{1}[a-zA-Z]{4}\d{4}',  # Pattern 2: char, char, digit, char, char, char, char, digit, digit, digit, digit  = 11
                r'[a-zA-Z]{2}\d{2}[a-zA-Z]{4}\d{4}',  # Pattern 2: char, char, digit, digit, char, char, char, char, digit, digit, digit, digit  = 12
                r'[a-zA-Z]{2}\d{1}[a-zA-Z]{5}\d{4}',  # Pattern 2: char, char, digit, char, char, char, char, char, digit, digit, digit, digit  = 12
                r'[a-zA-Z]{2}\d{1}[a-zA-Z]{5}\d{4}',  # Pattern 2: char, char, digit, char, char, char, char, char, digit, digit, digit, digit  = 12
                r'[a-zA-Z]{2}\d{2}[a-zA-Z]{5}\d{4}',  # Pattern 2: char, char, digit, digit, char, char, char, char, char, digit, digit, digit, digit  = 13
                r'[a-zA-Z]{2}\d{1}[a-zA-Z]{6}\d{4}',  # Pattern 2: char, char, digit, char, char, char, char, char, char, digit, digit, digit, digit  = 13
                r'\d{2}[a-zA-Z]{2}\d{4}[a-zA-Z]',      # digit, digit, char, char, digit, digit, digit, digit,char         
                ]
                
                # Only proceed if the length of the detected text is valid
                if 8 <= length <= 13:
                    # Find matches for the plate pattern
                    all_matches = [match for pattern in patterns for match in re.findall(pattern, t)]
                    
                    if all_matches:
                        for match in all_matches:
                            # Only save if this plate is different from the last detected one
                            match = match.upper()
                            if match not in temp:
                                temp.append(match)  # Update the temp list with the new plate text
                                print("plate text :", match)
                                return match
    clear_temp_after_interval()
    return None                                
def process_video(camera_url, frame_interval, retry_limit=50):
    """
    Process the video stream and send frames to RabbitMQ.
    """
    # Load ONNX model
    session = ort.InferenceSession("license_plate_detector.onnx")
    retry_count = 0
    try:
        camera_url = int(camera_url)
    except ValueError:
        camera_url = camera_url
    while retry_count < retry_limit:
        
        cap = cv2.VideoCapture(camera_url)

        if not cap.isOpened():
            print(f"Error: Could not open video stream from {camera_url}")
            retry_count += 1
            time.sleep(5)
            continue
        
        frame_count = 0
        last_frame_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    if time.time() - last_frame_time > 5:
                        print(f"No frame received for 5 seconds from {camera_url}, restarting...")
                        break
                    continue

                last_frame_time = time.time()

                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue

                frame_count = 0
                orig_h, orig_w = frame.shape[:2]
                # Preprocess
                input_tensor = preprocess_frame(frame)

                # Run inference
                outputs = session.run(None, {"images": input_tensor})

                # Postprocess
                boxes, confidences = postprocess(outputs, (orig_h, orig_w))
                # Draw results
                frame = draw_boxes(frame, boxes, confidences)
                # Crop license plates from frame
                plates = crop_plates(frame, boxes)

                # Optional: Show cropped plates in separate windows
                for i, plate_img in enumerate(plates):
                    if plate_img.size != 0:
                        # plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                        # _, plate_crop_thresh = cv2.threshold(plate_gray, 155, 255, cv2.THRESH_BINARY_INV)
                        cv2.imshow(f"Plate {i+1}", plate_img)
                        plate_text=read_license_plate(plate_img)
                        print("This is return text :", plate_text)


                # Show frame
                cv2.imshow("License Plate Detection - Press 'q' to quit", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        except Exception as e:
            print(f"An error occurred in camera : {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            retry_count += 1
            if retry_count >= retry_limit:
                print(f"Failed to process video stream after {retry_count} retries.")
                break

if __name__ == "__main__":
    # Fetch camera ID and RTSP URL from RabbitMQ queue 'details'
    process_video(camera_url="0", frame_interval=1)
