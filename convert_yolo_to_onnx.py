from ultralytics import YOLO




# # ---------------------load the model and chech the number of classes----------------
# model=YOLO("license_plate_detector.pt")
# # check the number of classes
# print("Number of class :", model.names)
# print("Model input shape:", model.model.args['imgsz'])
# # ---------------------load the model and chech the number of classes----------------



# ---------------------load the model and export to onnx for dynamic size ----------------
model = YOLO("license_plate_detector.pt")

# Export to ONNX
model.export(format="onnx", opset=12, dynamic=True)

print("ONNX export complete.")
# ---------------------load the model and export to onnx for dynamic size ----------------