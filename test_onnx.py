import torch
import onnxruntime as ort
import numpy as np
from src.model import build_mobilenet_v3

# 1. Load your original PyTorch model
num_classes = 15
pytorch_model = build_mobilenet_v3(num_classes=num_classes)
pytorch_model.load_state_dict(torch.load("models/best_mobilenetv3.pth", map_location=torch.device('cpu')))
pytorch_model.eval()

# 2. Create the exact same dummy input used for export
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Get PyTorch prediction
with torch.no_grad():
    pytorch_output = pytorch_model(dummy_input).numpy()

# 4. Load the ONNX model
ort_session = ort.InferenceSession("mobilenetv3_cropcare.onnx")

# 5. Get ONNX prediction
ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
onnx_output = ort_session.run(None, ort_inputs)[0]

# 6. Compare the raw confidence scores
print("PyTorch Output (first 5):", pytorch_output[0][:5])
print("ONNX Output (first 5):   ", onnx_output[0][:5])

try:
    np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-03, atol=1e-05)
    print("\n✅ Success! ONNX model matches PyTorch predictions perfectly.")
except AssertionError as e:
    print("\n❌ Error: Model outputs do not match.")
    print(e)