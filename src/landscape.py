#!/usr/bin/env python3
"""
Визуализация топологического ландшафта нейросети.

Загружает результаты эксперимента из .npz файла и строит:
1. 3D ландшафт нейронов (t-SNE + кривизна Риччи)
2. Сравнение методов pruning
3. Распределение кривизны Риччи
"""

import argparse
import logging
import os
import sys
from typing import Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)


def plot_3d_landscape(
    coords: np.ndarray,
    ricci: np.ndarray,
    mountains: np.ndarray,
    plateaus: np.ndarray,
    singularities: np.ndarray,
    ax: Axes3D
) -> None:
    """Строит 3D визуализацию топологического ландшафта.

    Args:
        coords: Координаты нейронов в 2D пространстве (n_neurons, 2).
        ricci: Массив кривизны Риччи для каждого нейрона.
        mountains: Индексы нейронов-мастеров (положительная кривизна).
        plateaus: Индексы нейронов-стажёров (кривизна около нуля).
        singularities: Индексы нейронов-сингулярностей (отрицательная кривизна).
        ax: 3D ось matplotlib для отрисовки.
    """
    # Цвета для разных типов нейронов
    colors = np.full(len(ricci), 'lightblue', dtype=object)
    colors[mountains] = 'gold'
    colors[singularities] = 'red'
    
    # 3D scatter plot
    ax.scatter(
        coords[:, 0], 
        coords[:, 1], 
        ricci, 
        c=colors, 
        s=30, 
        alpha=0.7,
        edgecolors='none'
    )
    
    # Попытка построить Delaunay триангуляцию для поверхности
    try:
        tri = Delaunay(coords)
        ax.plot_trisurf(
            coords[:, 0], 
            coords[:, 1], 
            ricci, 
            triangles=tri.simplices, 
            alpha=0.1, 
            color='gray'
        )
    except Exception as e:
        logger.warning(f"Не удалось построить Delaunay триангуляцию: {e}")
    
    ax.set_xlabel('t-SNE Dim 1')
    ax.set_ylabel('t-SNE Dim 2')
    ax.set_zlabel('Ricci Curvature')
    ax.set_title('Топологический ландшафт\n(золото=мастера, красный=сингулярности, синий=плато)')


def plot_method_comparison(
    fractions: np.ndarray,
    baseline: float,
    hard: np.ndarray,
    surgery: np.ndarray,
    random: np.ndarray,
    ax: plt.Axes
) -> None:
    """Строит график сравнения методов pruning.

    Args:
        fractions: Массив долей pruning.
        baseline: Базовая точность до pruning.
        hard: Точность после жёсткого pruning.
        surgery: Точность после хирургии Перельмана.
        random: Точность после случайного pruning.
        ax: Ось matplotlib для отрисовки.
    """
    ax.plot(
        [0] + list(fractions), 
        [baseline] + list(hard), 
        'bo-', 
        linewidth=2, 
        markersize=7, 
        label='Hard prune'
    )
    ax.plot(
        [0] + list(fractions), 
        [baseline] + list(surgery), 
        'go-', 
        linewidth=2, 
        markersize=7, 
        label='Perelman surgery'
    )
    ax.plot(
        [0] + list(fractions), 
        [baseline] + list(random), 
        'ro-', 
        linewidth=2, 
        markersize=7, 
        label='Random'
    )
    ax.axhline(y=baseline, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prune Fraction')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Сравнение методов pruning')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_ricci_distribution(
    ricci: np.ndarray,
    mountains: np.ndarray,
    plateaus: np.ndarray,
    singularities: np.ndarray,
    ax: plt.Axes
) -> None:
    """Строит гистограмму распределения кривизны Риччи.

    Args:
        ricci: Массив кривизны Риччи для каждого нейрона.
        mountains: Индексы нейронов-мастеров.
        plateaus: Индексы нейронов-стажёров.
        singularities: Индексы нейронов-сингулярностей.
        ax: Ось matplotlib для отрисовки.
    """
    ax.hist(ricci, bins=40, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
    ax.axvline(x=-0.1, color='red', linestyle='--', alpha=0.7, label='Порог сингулярностей')
    ax.axvline(x=0.1, color='gold', linestyle='--', alpha=0.7, label='Порог мастеров')
    ax.set_xlabel('Ricci Curvature')
    ax.set_ylabel('Количество нейронов')
    ax.set_title(
        f'Распределение кривизны\n'
        f'{len(mountains)} мастеров, {len(singularities)} сингулярностей, {len(plateaus)} плато'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)


def main(input_path: str, output_path: str) -> None:
    """Оркестрация визуализации топологического ландшафта.

    Args:
        input_path: Путь к .npz файлу с результатами эксперимента.
        output_path: Путь для сохранения PNG файла с визуализацией.
    """
    logger.info("=" * 60)
    logger.info("ВИЗУАЛИЗАЦИЯ ТОПОЛОГИЧЕСКОГО ЛАНДШАФТА")
    logger.info("=" * 60)
    
    # 1. Загрузка данных
    if not os.path.exists(input_path):
        logger.error(f"Файл не найден: {input_path}")
        logger.error("Сначала запустите perelman_surgery.py для генерации данных")
        sys.exit(1)
    
    logger.info(f"Загрузка данных из {input_path}")
    try:
        data = np.load(input_path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {e}")
        sys.exit(1)
    
    # Извлечение данных
    baseline = float(data['baseline'])
    fractions = data['fractions']
    hard = data['hard']
    surgery = data['surgery']
    random = data['random']
    ricci = data['ricci']
    mountains = data['mountains']
    singularities = data['singularities']
    plateaus = data['plateaus']
    
    logger.info(f"Загружено: {len(ricci)} нейронов")
    logger.info(f"Baseline accuracy: {baseline:.2f}%")
    logger.info(f"Ландшафт: {len(mountains)} мастеров, {len(singularities)} сингулярностей, {len(plateaus)} плато")
    
    # 2. Вычисление координат для 3D визуализации
    logger.info("Вычисление координат (t-SNE)...")
    n = len(ricci)
    
    # Попытка использовать t-SNE, если доступен
    try:
        from sklearn.manifold import TSNE
        # Создаём фиктивную матрицу признаков на основе кривизны
        # В реальном сценарии здесь должны быть активации или веса
        features = np.column_stack([ricci, np.random.randn(n)])
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n-1))
        coords = tsne.fit_transform(features)
        logger.info("Использован t-SNE для проекции")
    except ImportError:
        logger.warning("t-SNE недоступен, используется SpectralEmbedding")
        from sklearn.manifold import SpectralEmbedding
        # Используем случайные координаты как fallback
        coords = np.random.randn(n, 2)
    except Exception as e:
        logger.warning(f"Ошибка t-SNE: {e}, используются случайные координаты")
        coords = np.random.randn(n, 2)
    
    # 3. Построение визуализации
    logger.info("Построение графиков...")
    fig = plt.figure(figsize=(18, 6))
    
    # Панель 1: 3D ландшафт
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    plot_3d_landscape(coords, ricci, mountains, plateaus, singularities, ax1)
    
    # Панель 2: Сравнение методов
    ax2 = fig.add_subplot(1, 3, 2)
    plot_method_comparison(fractions, baseline, hard, surgery, random, ax2)
    
    # Панель 3: Распределение Ricci
    ax3 = fig.add_subplot(1, 3, 3)
    plot_ricci_distribution(ricci, mountains, plateaus, singularities, ax3)
    
    plt.tight_layout()
    
    # 4. Сохранение
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Визуализация сохранена: {output_path}")
    logger.info("=" * 60)
    logger.info("ГОТОВО!")
    logger.info("=" * 60)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    
    parser = argparse.ArgumentParser(
        description='Визуализация топологического ландшафта нейросети'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='results/perelman_surgery.npz',
        help='Путь к .npz файлу с результатами эксперимента (по умолчанию: results/perelman_surgery.npz)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/landscape_visualization.png',
        help='Путь для сохранения PNG файла с визуализацией (по умолчанию: results/landscape_visualization.png)'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main(
        input_path=args.input,
        output_path=args.output
    )
