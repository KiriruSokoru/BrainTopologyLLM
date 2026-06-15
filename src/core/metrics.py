import torch
from typing import Tuple
from torch.utils.data import DataLoader

__all__ = [
    "accuracy",
    "evaluate",
    "train_model",
    "train_epoch",
]

def accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Вычисляет долю верных предсказаний модели на датасете.

    Args:
        model: Модель PyTorch.
        loader: DataLoader с тестовыми данными.
        device: Устройство вычислений.

    Returns:
        Точность в диапазоне [0.0, 1.0].
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total if total > 0 else 0.0


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Вычисляет среднюю потерю и точность модели на датасете.

    Args:
        model: Модель PyTorch.
        loader: DataLoader с данными.
        device: Устройство вычислений.

    Returns:
        Кортеж (loss, accuracy).
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    avg_loss = running_loss / total if total > 0 else float("inf")
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 0.001
) -> float:
    """Полный цикл обучения модели.

    Args:
        model: Модель PyTorch.
        train_loader: DataLoader с обучающими данными.
        epochs: Количество эпох обучения.
        device: Устройство вычислений.
        lr: Скорость обучения.

    Returns:
        Финальная точность (accuracy) на обучающей выборке.
    """
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    final_acc = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        final_acc = correct / total if total > 0 else 0.0
        
    return final_acc


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Train model for a single epoch and return average loss as float.

    Args:
        model: PyTorch model.
        loader: DataLoader providing training batches.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Computation device.

    Returns:
        Average loss over the epoch as float.
    """
    model.train()
    model.to(device)
    running_loss = 0.0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

    avg_loss = running_loss / total if total > 0 else 0.0
    return float(avg_loss)
