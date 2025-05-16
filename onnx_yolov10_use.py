import cv2
import numpy as np
import onnxruntime as ort

# Load class names
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Load ONNX model
model_path = "yolov10n.onnx"
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
# session = ort.InferenceSession("yolov10n.onnx", providers=['CPUExecutionProvider'])
# session.set_providers(['CPUExecutionProvider'], [{'use_cpu_mem_arena': True}])

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1, 3, H, W]

# Preprocessing
def preprocess(image, input_shape):
    h, w = input_shape[2], input_shape[3]
    img = cv2.resize(image, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    return img

# Postprocessing
def postprocess(outputs, original_shape, input_shape, conf_thres=0.3):
    pred = outputs[0][0]  # [num_boxes, 6]
    orig_h, orig_w = original_shape[:2]
    inp_h, inp_w = input_shape[2], input_shape[3]
    scale_w, scale_h = orig_w / inp_w, orig_h / inp_h

    boxes, scores, class_ids = [], [], []
    for x1, y1, x2, y2, conf, cls_id in pred:
        if conf < conf_thres:
            continue
        box = [
            int(x1 * scale_w),
            int(y1 * scale_h),
            int(x2 * scale_w),
            int(y2 * scale_h)
        ]
        boxes.append(box)
        scores.append(float(conf))
        class_ids.append(int(cls_id))

    return boxes, scores, class_ids

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame, input_shape)
    outputs = session.run(None, {input_name: input_tensor})
    boxes, scores, class_ids = postprocess(outputs, frame.shape, input_shape)

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        label = f"{class_names[cls_id]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("ONNX YOLOv10 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
