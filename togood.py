# export_pinn_to_onnx.py
# Loads your trained .pth model and exports it to ONNX

import torch
import torch.nn as nn

# ────────────────────────────────────────────────
# 1. Define EXACT same model class as in training
# ────────────────────────────────────────────────

class SolarPINN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Learnable degradation rate — must be included
        self.lambda_deg = nn.Parameter(torch.tensor(3e-5))  # dummy value, will be overwritten

    def forward(self, x):
        return self.net(x)

# ────────────────────────────────────────────────
# 2. Load the trained weights
# ────────────────────────────────────────────────

MODEL_PATH = "best_pinn_model_pr_raw.pth"     # your saved .pth file
ONNX_PATH  = "solar_pinn_model.onnx"

model = SolarPINN(input_dim=10)               # ← 10 = number of input features
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()                                  # IMPORTANT: switch to evaluation mode

print("Model loaded successfully from .pth")

# ────────────────────────────────────────────────
# 3. Create dummy input (must match training shape)
#    shape = (batch_size=1, features=10)
# ────────────────────────────────────────────────

dummy_input = torch.randn(1, 10, dtype=torch.float32)

# ────────────────────────────────────────────────
# 4. Export to ONNX
# ────────────────────────────────────────────────

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,                # include trained weights
    opset_version=17,                  # 13–18 are safe; 17 is good balance
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model successfully exported to: {ONNX_PATH}")

# Optional: quick validation with onnx
try:
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed ✓")
except ImportError:
    print("onnx package not installed — install with: pip install onnx")
except Exception as e:
    print("ONNX check failed:", e)