import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader
import torch.nn as nn

def collect_activations(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int = 50) -> Dict[str, np.ndarray]:
    """Собирает активации слоёв модели на батчах из датасета.

    Args:
        model: Модель PyTorch.
        loader: DataLoader для сбора данных.
        device: Устройство вычислений.
        max_batches: Максимальное количество батчей для обработки.

    Returns:
        Словарь {layer_name: np.ndarray}, где каждый массив имеет форму (n_samples, n_neurons).
        Для свёрточных слоёв пространственные размерности усреднены.
    """
    store: Dict[str, List[np.ndarray]] = {name: [] for name, _ in model.named_modules()}
    hooks = []

    def _hook_fn(name: str):
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            # Усредняем spatial dims для Conv2d: [N, C, H, W] -> [N, C]
            if output.dim() == 4:
                out = output.mean(dim=[2, 3])
            else:
                out = output
            store[name].append(out.detach().cpu().numpy())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(_hook_fn(name)))

    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= max_batches:
                break
            _ = model(inputs.to(device))

    for h in hooks:
        h.remove()

    # Конкатенируем батчи в единый массив по оси сэмплов
    result: Dict[str, np.ndarray] = {}
    for name, batch_list in store.items():
        if batch_list:
            result[name] = np.concatenate(batch_list, axis=0)
        else:
            result[name] = np.empty((0, 0))

    return result
