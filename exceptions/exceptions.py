class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""
    def __init__(self, message="Invalid backbone architecture. "
                               "Check the config file and pass one of: alexnet, resnet18 or resnet50"):
        self.message = message
        super().__init__(self.message)


class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""


class InvalidCheckpointPath(BaseSimCLRException):
    """Raised when the path of the checkpoint is invalid"""
