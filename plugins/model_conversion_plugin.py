
from plugin_base import PluginBase
import torch
import os
from plugins.starter_train_plugin import SimpleNet

class ModelConversionPlugin(PluginBase):
    def run(self, params: dict) -> dict:
        input_path = params.get("input", "models/model.pth")
        output_path = params.get("output", "models/model.onnx")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Model file not found: {input_path}")

        # Load model architecture and weights
        model = SimpleNet(input_size=20, hidden_size=64, num_classes=2)
        model.load_state_dict(torch.load(input_path))
        model.eval()

        # Create dummy input for export
        dummy_input = torch.randn(1, model.fc1.in_features)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        return {"status": "success", "details": f"Converted to ONNX at {output_path}"}