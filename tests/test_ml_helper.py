import os
import sys

# ensure project root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import json

from cli import parse_task
from nlp_classifier import classify_task
from plugin_factory import get_plugin
from plugins.visualization_plugin import VisualizationPlugin
from plugins.starter_train_plugin import StarterTrainPlugin
from plugins.model_conversion_plugin import ModelConversionPlugin

# --- Test classification ---

def test_classify_visualization_default():
    task = "build ROC curve from metrics.json"
    task_type, params = classify_task(task)
    assert task_type == "visualization"
    assert params["file"] == "metrics.json"


def test_classify_train_with_params():
    task = "train model epochs=10 batch_size=16"
    task_type, params = classify_task(task)
    assert task_type == "starter-train"
    assert params.get("epochs") == 10
    assert params.get("batch_size") == 16


def test_classify_convert_paths():
    task = "convert path/to/model.pth to export/model.onnx"
    task_type, params = classify_task(task)
    assert task_type == "model-conversion"
    assert params["input"] == "path/to/model.pth"
    assert params["output"] == "export/model.onnx"

# --- Test CLI delegation ---

def test_cli_parse_delegates_to_classifier(monkeypatch):
    import cli
    monkeypatch.setattr(cli, 'classify_task', lambda x: ("foo", {"bar": 1}))
    task_type, params = parse_task("any input")
    assert task_type == "foo"
    assert params == {"bar": 1}

# --- Test plugin factory ---

def test_get_plugin_valid():
    plugin = get_plugin("visualization")
    assert isinstance(plugin, VisualizationPlugin)
    plugin = get_plugin("starter-train")
    assert isinstance(plugin, StarterTrainPlugin)
    plugin = get_plugin("model-conversion")
    assert isinstance(plugin, ModelConversionPlugin)


def test_get_plugin_invalid():
    with pytest.raises(ValueError):
        get_plugin("nonexistent")

# --- Test plugins ---

def test_visualization_plugin_run(tmp_path):
    # Create sample metrics.json
    data = {"fpr": [0, 1], "tpr": [0, 1]}
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps(data))

    plugin = VisualizationPlugin()
    # Override output_dir to tmp_path for isolation
    result = plugin.run({"file": str(metrics_file), "output_dir": str(tmp_path)})

    assert result["status"] == "success"
    output_path = result["output_path"]
    # Check file exists
    assert os.path.exists(output_path)
    assert output_path.startswith(str(tmp_path))
    # AUC for straight line is 0.5
    assert pytest.approx(result["auc"], rel=1e-3) == 0.5


def test_conversion_plugin_file_not_found(tmp_path):
    plugin = ModelConversionPlugin()
    with pytest.raises(FileNotFoundError):
        plugin.run({"input": str(tmp_path / "does_not_exist.pth"), "output": "out.onnx"})
