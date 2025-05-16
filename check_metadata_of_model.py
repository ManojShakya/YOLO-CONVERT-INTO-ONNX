# import onnx

# # Load model
# model = onnx.load("license_plate_detector.onnx")

# # Print model outputs
# print("== Output Info ==")
# for output in model.graph.output:
#     print(f"Name: {output.name}")
#     for dim in output.type.tensor_type.shape.dim:
#         print(f" - dim: {dim.dim_value if dim.HasField('dim_value') else '?'}")





# import onnxruntime as ort
# import numpy as np

# # Load ONNX model
# session = ort.InferenceSession("license_plate_detector.onnx")

# # Run with dummy input
# dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
# outputs = session.run(None, {"images": dummy_input})

# output_tensor = outputs[0]
# print("ONNX Output Shape:", output_tensor.shape)


import onnxruntime as ort
import numpy as np

# Load the exported ONNX model
session = ort.InferenceSession("license_plate_detector.onnx")

# Prepare dummy input
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Run inference
outputs = session.run(None, {"images": dummy_input})
print("ONNX inference successful ✅. Output shape:", [o.shape for o in outputs])
