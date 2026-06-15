import warnings
import numpy as np
import networkx as nx

def compute_ricci_curvature(graph: nx.Graph, alpha: float = 0.5) -> np.ndarray:
    """Вычисляет кривизну Оливье-Риччи для графа активаций.

    Args:
        graph: NetworkX граф с атрибутами веса рёбер.
        alpha: Параметр распределения массы для вычисления кривизны (по умолчанию 0.5).

    Returns:
        Массив кривизн для каждого узла графа (усреднённый по инцидентным рёбрам).
    """
    from GraphRicciCurvature.OllivierRicci import OllivierRicci

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orc = OllivierRicci(graph, alpha=alpha)
        orc.compute_ricci_curvature()

    # GraphRicciCurvature сохраняет кривизну рёбер в атрибуте 'ricciCurvature'
    node_curvatures = {}
    for node in graph.nodes():
        edges = graph.edges(node, data=True)
        if edges:
            weights = [data.get("ricciCurvature", 0.0) for _, _, data in edges]
            node_curvatures[node] = np.mean(weights)
        else:
            node_curvatures[node] = 0.0

    return np.array([node_curvatures[n] for n in sorted(graph.nodes())])
