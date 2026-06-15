import numpy as np
import networkx as nx
from typing import Dict, Tuple, List

def build_activation_graph(activations: Dict[str, np.ndarray], corr_threshold: float = 0.7) -> Tuple[nx.Graph, np.ndarray, List[str]]:
    """Строит граф корреляций на основе собранных активаций нейронов.

    Args:
        activations: Словарь {layer_name: np.ndarray} с активациями формы (n_samples, n_neurons).
        corr_threshold: Порог корреляции Пирсона для создания ребра.

    Returns:
        Кортеж (networkx.Graph, матрица смежности корреляций, список меток нейронов).
    """
    layer_names = sorted(activations.keys())
    neuron_labels: List[str] = []
    activation_chunks: List[np.ndarray] = []

    for name in layer_names:
        arr = activations[name]
        if arr.size == 0:
            continue
        n_neurons = arr.shape[1]
        neuron_labels.extend([f"{name}_n{i}" for i in range(n_neurons)])
        activation_chunks.append(arr)

    if not activation_chunks:
        raise ValueError("Не удалось собрать валидные активации для построения графа.")

    X = np.concatenate(activation_chunks, axis=1)
    corr_matrix = np.corrcoef(X.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    G = nx.Graph()
    n = len(neuron_labels)
    for i in range(n):
        G.add_node(i, label=neuron_labels[i])
        for j in range(i + 1, n):
            if corr_matrix[i, j] >= corr_threshold:
                G.add_edge(i, j, weight=float(corr_matrix[i, j]))

    return G, corr_matrix, neuron_labels

def parse_neuron_labels(labels: List[str]) -> Dict[int, Tuple[str, int]]:
    """Парсит метки нейронов в словарь маппинга индексов.

    Args:
        labels: Список меток вида "layer_name:neuron_idx".

    Returns:
        Словарь {глобальный_индекс: (имя_слоя, локальный_индекс)}.
    """
    mapping: Dict[int, Tuple[str, int]] = {}
    for idx, label in enumerate(labels):
        parts = label.split(':')
        if len(parts) == 2:
            layer, neuron = parts[0], int(parts[1])
        else:
            layer, neuron = parts[0], 0  # Fallback, если индекс не указан
        mapping[idx] = (layer, neuron)
    return mapping    
