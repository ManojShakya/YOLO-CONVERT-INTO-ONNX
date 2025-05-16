import cv2
import numpy as np
import onnxruntime as ort

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

def main():
    # Load ONNX model
    session = ort.InferenceSession("license_plate_detector.onnx")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        input_tensor = preprocess_frame(frame)

        # Run inference
        outputs = session.run(None, {"images": input_tensor})

        # Postprocess
        boxes, confidences = postprocess(outputs, (orig_h, orig_w))

        # Draw results
        frame = draw_boxes(frame, boxes, confidences)

        # Show frame
        cv2.imshow("License Plate Detection - Press 'q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
