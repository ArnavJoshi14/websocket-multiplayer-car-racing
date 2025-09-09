import torch

SEQ_LEN = 20
STATE_DIM = 5  # [posX, posY, posZ, speed, rot]

# Load TorchScript model
ts_model = torch.jit.load("lstm_model.pt")
ts_model.eval()

# Dummy input with the same shape as training
dummy_input = torch.randn(1, SEQ_LEN, STATE_DIM)

# Export to ONNX
torch.onnx.export(
    ts_model,
    dummy_input,
    "lstm_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size"}
    },
    opset_version=17
)

print("Model exported to lstm_model.onnx")
