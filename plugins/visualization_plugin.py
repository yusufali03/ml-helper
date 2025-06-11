# plugins/visualization_plugin.py

from plugin_base import PluginBase
import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class VisualizationPlugin(PluginBase):
    def run(self, params: dict) -> dict:
        file_path = params.get("file")
        output_dir = params.get("output_dir", "outputs")

        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Metrics file not found: {file_path}")

        # Load metrics data
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Determine curve type
        is_roc = "fpr" in data and "tpr" in data
        is_pr = "precision" in data and "recall" in data
        if is_roc:
            x = data["fpr"]
            y = data["tpr"]
            xlabel = "False Positive Rate"
            ylabel = "True Positive Rate"
            title = "ROC Curve"
            # Compute AUC
            score = float(np.trapz(y, x))
            label = f"AUC = {score:.3f}"
        elif is_pr:
            x = data["recall"]
            y = data["precision"]
            xlabel = "Recall"
            ylabel = "Precision"
            title = "Precision-Recall Curve"
            # Compute area under curve (Average Precision approximation)
            score = float(np.trapz(y, x))
            label = f"AP = {score:.3f}"
        else:
            raise ValueError("JSON must contain either 'fpr'/'tpr' or 'precision'/'recall' arrays.")

        # Compute timestamped filename
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        curve = "roc" if is_roc else "pr"
        filename = f"{curve}_{ts}.png"
        output_path = os.path.join(output_dir, filename)

        # Plot
        plt.figure()
        plt.plot(x, y, label=label)
        if is_roc:
            plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return {
            "status": "success",
            "details": f"{title} saved to {output_path}",
            "output_path": output_path,
            "score": score,
            "curve_type": "roc" if is_roc else "pr"
        }
