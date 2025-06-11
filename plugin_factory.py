# plugin_factory.py

from plugins.visualization_plugin import VisualizationPlugin
from plugins.starter_train_plugin import StarterTrainPlugin
from plugins.model_conversion_plugin import ModelConversionPlugin

PLUGIN_REGISTRY = {
    "visualization": VisualizationPlugin,
    "starter-train": StarterTrainPlugin,
    "model-conversion": ModelConversionPlugin,
}

def get_plugin(task_type: str):
    """
    Return an instance of the plugin class matching task_type.
    Raises ValueError if no matching plugin.
    """
    plugin_cls = PLUGIN_REGISTRY.get(task_type)
    if not plugin_cls:
        raise ValueError(f"No plugin found for task type '{task_type}'")
    return plugin_cls()
