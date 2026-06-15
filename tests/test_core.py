"""
Модульные тесты для компонентов src.core.

Используются фиктивные данные и моки для обеспечения изолированности 
и скорости выполнения тестов без загрузки реальных моделей или датасетов.
"""

import pytest
import torch
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader
from unittest.mock import MagicMock, patch

from src.core.models import MNIST_CNN
from src.core.metrics import accuracy
from src.core.graph_builder import build_activation_graph


class TestMNISTCNN:
    """Тесты для модели MNIST_CNN."""

    def test_mnist_cnn_forward(self) -> None:
        """Проверяет, что MNIST_CNN принимает вход (1, 1, 28, 28) и возвращает (1, 10)."""
        model = MNIST_CNN()
        # Создаем фиктивный батч из 1 изображения 28x28 с 1 каналом
        dummy_input = torch.randn(1, 1, 28, 28)
        
        output = model(dummy_input)
        
        assert output.shape == (1, 10), f"Ожидалась форма (1, 10), получена {output.shape}"

    def test_mnist_cnn_hooks(self) -> None:
        """Проверяет, что после forward-pass в словаре activations появляются ключи слоев."""
        model = MNIST_CNN()
        dummy_input = torch.randn(1, 1, 28, 28)
        
        _ = model(dummy_input)
        
        # Проверяем, что хуки сработали и записали активации
        assert len(model.activations) > 0, "Словарь activations пуст после forward-pass"
        
        # Проверяем наличие ожидаемых слоев (согласно реализации в src/core/models.py)
        expected_layers = {"conv1", "conv2", "fc1", "fc2"}
        assert expected_layers.issubset(model.activations.keys()), \
            f"Ожидаемые слои {expected_layers} не найдены в {model.activations.keys()}"
        
        # Проверяем, что активации — это тензоры PyTorch
        for layer_name, activation in model.activations.items():
            assert isinstance(activation, torch.Tensor), f"Активация слоя {layer_name} не является torch.Tensor"


class TestMetrics:
    """Тесты для функций метрик."""

    def test_accuracy_with_mock(self) -> None:
        """Проверяет корректность расчета accuracy с использованием мок-модели и мок-загрузчика."""
        # Создаем мок-модель, которая всегда возвращает предсказуемые логиты
        mock_model = MagicMock()
        # Логиты: для 2-х сэмплов модель уверенно предсказывает класс 0 и класс 1 соответственно
        mock_model.return_value = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        
        # Создаем мок-датасет
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        
        inputs = torch.randn(2, 5)
        targets = torch.tensor([0, 1]) # Целевые классы совпадают с предсказаниями модели
        
        # Настраиваем __getitem__ для возврата кортежа (input, target)
        mock_dataset.__getitem__.side_effect = lambda i: (inputs[i], targets[i])
        
        loader = DataLoader(mock_dataset, batch_size=2)
        
        # Вычисляем точность
        acc = accuracy(mock_model, loader, device=torch.device("cpu"))
        
        # Ожидаем 100% точность (1.0), так как предсказания идеальны
        assert acc == 1.0, f"Ожидалась точность 1.0, получена {acc}"
        
        # Проверяем, что модель была вызвана ровно один раз (один батч)
        mock_model.assert_called_once()

    def test_accuracy_imperfect_mock(self) -> None:
        """Проверяет расчет accuracy при частичном совпадении предсказаний."""
        mock_model = MagicMock()
        # Модель предсказывает класс 0 для обоих сэмплов
        mock_model.return_value = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        inputs = torch.randn(2, 5)
        targets = torch.tensor([0, 1]) # Первый верный, второй неверный
        
        mock_dataset.__getitem__.side_effect = lambda i: (inputs[i], targets[i])
        loader = DataLoader(mock_dataset, batch_size=2)
        
        acc = accuracy(mock_model, loader, device=torch.device("cpu"))
        
        # Ожидаем 50% точность (0.5)
        assert acc == 0.5, f"Ожидалась точность 0.5, получена {acc}"


class TestGraphBuilder:
    """Тесты для построения графа активаций."""

    def test_build_activation_graph(self) -> None:
        """Проверяет, что граф корректно строится на основе фиктивных активаций."""
        # Создаем фиктивные активации: 10 сэмплов, по 3 нейрона в каждом из 2 слоев
        # Делаем нейрон 0 и нейрон 1 в слое 1 идеально коррелирующими
        act_layer1 = np.array([
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 3.0, 0.0],
        ] * 4)  # Форма: (12, 3)
        
        # Слой 2: случайный шум (низкая корреляция)
        np.random.seed(42)
        act_layer2 = np.random.randn(12, 3)
        
        mock_activations = {
            "layer1": act_layer1,
            "layer2": act_layer2
        }
        
        # Строим граф с высоким порогом корреляции (0.8), чтобы поймать только идеальную связь
        graph, adj_matrix, labels = build_activation_graph(mock_activations, corr_threshold=0.8)
        
        # Проверка типов и структур
        assert isinstance(graph, nx.Graph), "Результат должен быть экземпляром nx.Graph"
        assert isinstance(adj_matrix, np.ndarray), "Матрица смежности должна быть numpy массивом"
        assert isinstance(labels, list), "Метки должны быть списком"
        
        # Всего нейронов: 3 (layer1) + 3 (layer2) = 6
        assert len(labels) == 6, f"Ожидалось 6 меток, получено {len(labels)}"
        assert adj_matrix.shape == (6, 6), f"Ожидалась матрица 6x6, получена {adj_matrix.shape}"
        
        # Проверяем наличие ребра между идеально коррелирующими нейронами (индексы 0 и 1)
        assert graph.has_edge(0, 1), "Ребро между коррелирующими нейронами 0 и 1 не создано"
        assert graph[0][1]['weight'] >= 0.99, "Вес ребра между коррелирующими нейронами должен быть близок к 1.0"
        
        # Проверяем отсутствие ребер между шумовыми нейронами (порог 0.8)
        # Например, между нейроном 0 (layer1) и нейроном 3 (layer2_n0)
        assert not graph.has_edge(0, 3), "Не должно быть ребра между некоррелирующими нейронами разных слоев"
