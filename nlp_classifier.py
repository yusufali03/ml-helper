from typing import Tuple, Dict
import re

def classify_task(task_str: str) -> Tuple[str, Dict]:
    ts = task_str.lower()

    # Precision-Recall curve intent
    if any(keyword in ts for keyword in ["precision-recall", "precision recall", "pr curve", "precision", "recall"]):
        m = re.search(r"from\s+(\S+)", ts)
        file = m.group(1) if m else "metrics_pr.json"
        return "visualization", {"file": file}

    # — Visualization intent (ROC curve)
    if "roc" in ts:
        m = re.search(r"from\s+(\S+)", ts)
        file = m.group(1) if m else "metrics.json"
        return "visualization", {"file": file}

    # — Training intent
    elif "train" in ts:
        params: Dict[str, int] = {}
        # Optionally parse epochs and batch_size if present like "epochs=10"
        m_epochs = re.search(r"epochs?=(\d+)", ts)
        if m_epochs:
            params["epochs"] = int(m_epochs.group(1))
        m_bs = re.search(r"batch[_-]?size=(\d+)", ts)
        if m_bs:
            params["batch_size"] = int(m_bs.group(1))
        return "starter-train", params

    # — Model conversion intent
    elif "convert" in ts:
        inp = re.search(r"(\S+\.pth)", ts)
        out = re.search(r"to\s+(\S+\.onnx)", ts)
        input_path = inp.group(1) if inp else "models/model.pth"
        output_path = out.group(1) if out else "models/model.onnx"
        return "model-conversion", {"input": input_path, "output": output_path}

    # — Fallback
    else:
        raise ValueError(f"Could not classify task: '{task_str}'")
