# plugin_base.py

from abc import ABC, abstractmethod

class PluginBase(ABC):
    @abstractmethod
    def run(self, params: dict) -> dict:
        """
        Execute the plugin logic.

        :param params: parameters for the task
        :return: dict with at least keys "status" ("success"|"error") and optional "details"
        """
        pass
