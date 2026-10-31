import torch
from src.model import build_mobilenet_v3

# 1. Initialize the empty model architecture using your builder function
# IMPORTANT: Replace '4' with the actual number of classes in your dataset
# If you defined it in config.py, you can use: from src.config import Config -> Config.NUM_CLASSES
num_classes = 15
model = build_mobilenet_v3(num_classes=num_classes)

print("Loading best_mobilenetv3.pth...")
# 2. Load the trained weights
model.load_state_dict(torch.load("models/best_mobilenetv3.pth", map_location=torch.device('cpu')))
model.eval()

# 3. Create dummy input tensor
dummy_input = torch.randn(1, 3, 224, 224)

# 4. Export to ONNX
onnx_filename = "mobilenetv3_cropcare.onnx"
print(f"Exporting model to {onnx_filename}...")

torch.onnx.export(
    model,
    dummy_input,
    onnx_filename,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("ONNX Export Complete!")