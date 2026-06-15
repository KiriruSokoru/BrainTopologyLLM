import torch
import torch.nn as nn
from typing import Callable, Dict

__all__ = [
    "MNIST_CNN",
    "SimpleMLP",
]

class MNIST_CNN(nn.Module):
    """Базовая свёрточная сеть для классификации MNIST с поддержкой хуков активаций."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        
        # Контейнер для хранения активаций во время forward-pass
        self.activations: Dict[str, torch.Tensor] = {}
        self._hooks()

    def _hooks(self) -> None:
        """Регистрирует forward hooks на всех Conv2d и Linear слоях."""
        def _hook_fn(name: str) -> Callable:
            def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
                self.activations[name] = output.detach().cpu()
            return hook

        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.register_forward_hook(_hook_fn(name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleMLP(nn.Module):
    """Простая полносвязная сеть (MLP) для базовых экспериментов (например, MNIST)."""

    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)  # Flatten для MNIST
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
