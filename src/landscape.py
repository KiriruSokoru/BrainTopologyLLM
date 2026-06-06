#!/usr/bin/env python3
"""
Визуализация поверхности смысла (ландшафт кривизны Риччи).
Использует сохранённые данные из perelman_surgery.npz.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Загружаем данные
data = np.load('results/perelman_surgery.npz', allow_pickle=True)
ricci = data['ricci']
mountains = data['mountains']
singularities = data['singularities']
plateaus = data['plateaus']
baseline = float(data['baseline'])
fractions = data['fractions']
hard = data['hard']
surgery = data['surgery']
random = data['random']

n = len(ricci)

# Создаём координаты через спектральное вложение
from sklearn.manifold import SpectralEmbedding

# Строим матрицу смежности из сохранённых данных (упрощённо — из важности)
# Для визуализации используем t-SNE на ricci + importance
importance = np.abs(ricci) * (np.ones(n) * 5 + 1)  # упрощённо
features = np.column_stack([ricci, importance])

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n-1))
coords = tsne.fit_transform(features)

# Нормализуем координаты для красоты
coords[:, 0] = (coords[:, 0] - coords[:, 0].min()) / (coords[:, 0].max() - coords[:, 0].min())
coords[:, 1] = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())

# Создаём фигуру
fig = plt.figure(figsize=(20, 8))

# ==============================================================================
# Панель 1: 3D-ландшафт
# ==============================================================================
ax1 = fig.add_subplot(1, 3, 1, projection='3d')

# Разделяем по типам
for name, indices, color, marker, label in [
    ('mountains', mountains, '#FFD700', '^', 'Горы (мастера)'),
    ('plateaus', plateaus, '#87CEEB', 'o', 'Плато (стажёры)'),
    ('singularities', singularities, '#FF4444', 's', 'Сингулярности (хаотики)')
]:
    if len(indices) > 0:
        ax1.scatter(
            coords[indices, 0], coords[indices, 1], ricci[indices],
            c=color, marker=marker, s=40, alpha=0.8,
            edgecolors='white', linewidth=0.5, label=label
        )

# Рисуем связи между соседними точками
from scipy.spatial import Delaunay
if n > 3:
    tri = Delaunay(coords)
    for simplex in tri.simplices:
        for i in range(3):
            p1 = coords[simplex[i]]
            p2 = coords[simplex[(i+1)%3]]
            mid_ricci = (ricci[simplex[i]] + ricci[simplex[(i+1)%3]]) / 2
            ax1.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], [mid_ricci, mid_ricci],
                'gray', alpha=0.08, linewidth=0.3
            )

ax1.set_xlabel('Измерение 1', fontsize=10)
ax1.set_ylabel('Измерение 2', fontsize=10)
ax1.set_zlabel('Кривизна Риччи', fontsize=10)
ax1.set_title('Поверхность смысла\n', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.view_init(elev=25, azim=45)

# Добавляем плоскость нулевой кривизны
xx, yy = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
ax1.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

# ==============================================================================
# Панель 2: Сравнение методов
# ==============================================================================
ax2 = fig.add_subplot(1, 3, 2)

ax2.plot([0] + list(fractions), [baseline] + list(hard), 
         'o-', color='#2196F3', linewidth=2, markersize=7, label='Жёсткое удаление')
ax2.plot([0] + list(fractions), [baseline] + list(surgery), 
         's-', color='#4CAF50', linewidth=2, markersize=7, label='Хирургия Перельмана')
ax2.plot([0] + list(fractions), [baseline] + list(random), 
         '^-', color='#9E9E9E', linewidth=2, markersize=7, label='Случайное')

ax2.axhline(y=baseline, color='black', linestyle='--', alpha=0.5, linewidth=1)
ax2.fill_between([0, 0.3], baseline-2, baseline+2, alpha=0.1, color='green', label='±2% зона')

ax2.set_xlabel('Доля удалённых нейронов', fontsize=11)
ax2.set_ylabel('Точность (%)', fontsize=11)
ax2.set_title('Стабильность методов\n', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

# ==============================================================================
# Панель 3: Распределение с аннотациями
# ==============================================================================
ax3 = fig.add_subplot(1, 3, 3)

counts, bins, patches = ax3.hist(ricci, bins=35, color='steelblue', edgecolor='white', alpha=0.7)

# Раскрашиваем столбцы по зонам
for i, patch in enumerate(patches):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center < -0.1:
        patch.set_facecolor('#FF4444')
        patch.set_alpha(0.7)
    elif bin_center > 0.1:
        patch.set_facecolor('#FFD700')
        patch.set_alpha(0.7)
    else:
        patch.set_facecolor('#87CEEB')
        patch.set_alpha(0.7)

ax3.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
ax3.axvline(x=-0.1, color='#FF4444', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.axvline(x=0.1, color='#FFD700', linestyle='--', linewidth=1.5, alpha=0.7)

# Аннотации
ax3.annotate(f'Сингулярности\n{len(singularities)} нейронов', 
             xy=(-0.25, max(counts)*0.8), fontsize=10, color='#FF4444',
             ha='center', fontweight='bold')
ax3.annotate(f'Плато\n{len(plateaus)} нейронов', 
             xy=(0, max(counts)*0.5), fontsize=10, color='#4169E1',
             ha='center', fontweight='bold')
ax3.annotate(f'Горы\n{len(mountains)} нейронов', 
             xy=(0.5, max(counts)*0.8), fontsize=10, color='#B8860B',
             ha='center', fontweight='bold')

ax3.set_xlabel('Кривизна Риччи', fontsize=11)
ax3.set_ylabel('Число нейронов', fontsize=11)
ax3.set_title('Ландшафт сети\n', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Общий заголовок
fig.suptitle('Топологическая оптимизация нейросети: метод Перельмана\n'
             'MNIST CNN, 186 нейронов, кривизна Олливье-Риччи',
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('results/landscape_final.png', dpi=150, bbox_inches='tight')
print("Сохранено: results/landscape_final.png")

# Статистика для README
print(f"\nСтатистика ландшафта:")
print(f"  Всего нейронов: {n}")
print(f"  Горы (мастера): {len(mountains)} ({len(mountains)/n*100:.0f}%)")
print(f"  Плато (стажёры): {len(plateaus)} ({len(plateaus)/n*100:.0f}%)")
print(f"  Сингулярности (хаотики): {len(singularities)} ({len(singularities)/n*100:.0f}%)")
print(f"  Baseline точность: {baseline:.2f}%")
print(f"  Макс. точность surgery: {max(surgery):.2f}% при {fractions[np.argmax(surgery)]:.0%} pruning")

