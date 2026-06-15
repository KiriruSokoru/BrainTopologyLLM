import logging

__all__ = [
    "MNIST_CNN",
    "accuracy",
    "evaluate",
    "train_model",
    "collect_activations",
    "build_activation_graph",
    "compute_ricci_curvature",
    "parse_neuron_labels",
]

# Тихая инициализация логгера для предотвращения вывода в консоль при отсутствии настроек
logging.getLogger("brain_topology_llm.core").addHandler(logging.NullHandler())

from .models import MNIST_CNN
from .metrics import accuracy, evaluate, train_model
from .activations import collect_activations
from .graph_builder import build_activation_graph
from .ricci import compute_ricci_curvature
from .graph_builder import build_activation_graph, parse_neuron_labels
