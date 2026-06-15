# experiments/fundamentals/level0_single_neuron_ultimate.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class Level0_UltimatePrecision:
    """Один нейрон — предельная точность, идеально разделимые данные"""
    
    def __init__(self, n_runs=100, n_epochs=5000, fine_tune_epochs=2000):
        self.n_runs = n_runs
        self.n_epochs = n_epochs
        self.fine_tune_epochs = fine_tune_epochs
        self.results = []
        
    def generate_data(self, n_samples=1000):
        """Два идеально разделимых облака (без шума, но с отступом)"""
        # Создаём два облака с гарантированным отступом
        centers = [[-2, 0], [2, 0]]
        X, y = make_blobs(n_samples=n_samples, centers=centers, 
                          n_features=2, cluster_std=0.5, random_state=42)
        # Добавляем небольшой гарантированный отступ
        X[y==0, 0] += -0.5
        X[y==1, 0] += 0.5
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).reshape(-1, 1)
        return X, y
    
    def train_model(self, seed):
        """Обучение с анальным подходом к точности"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X, y = self.generate_data()
        model = nn.Linear(2, 1)
        
        # Этап 1: Грубая настройка
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        criterion = nn.BCEWithLogitsLoss()
        
        losses = []
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Этап 2: Тонкая настройка (очень маленький lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(self.fine_tune_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Этап 3: Сверхтонкая настройка (экспоненциально убывающий lr)
        for epoch in range(500):
            lr = 0.0001 * (0.99 ** epoch)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        final_loss = losses[-1]
        
        # Вычисляем точность
        with torch.no_grad():
            probs = torch.sigmoid(model(X))
            pred = (probs > 0.5).float()
            accuracy = (pred == y).float().mean().item()
        
        return {
            'seed': seed,
            'final_loss': final_loss,
            'accuracy': accuracy,
            'history': losses,
            'params': model.weight.data.numpy().copy(),
            'bias': model.bias.data.item()
        }
    
    def run_all(self):
        print("🔬 УРОВЕНЬ 0: ОДИН НЕЙРОН (ПРЕДЕЛЬНАЯ ТОЧНОСТЬ)")
        print("="*60)
        print(f"   Запусков: {self.n_runs}")
        print(f"   Эпох грубой настройки: {self.n_epochs}")
        print(f"   Эпох тонкой настройки: {self.fine_tune_epochs}")
        print("="*60)
        
        for seed in tqdm(range(self.n_runs), desc="Запуски"):
            result = self.train_model(seed)
            self.results.append(result)
        
        self.analyze()
    
    def analyze(self):
        final_losses = [r['final_loss'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        
        mean_loss = np.mean(final_losses)
        std_loss = np.std(final_losses)
        var_loss = np.var(final_losses)
        min_loss = np.min(final_losses)
        max_loss = np.max(final_losses)
        
        print(f"\n📊 РЕЗУЛЬТАТЫ (n={self.n_runs} запусков):")
        print(f"   {'='*50}")
        print(f"   Средний финальный loss: {mean_loss:.15f}")
        print(f"   Стандартное отклонение: {std_loss:.15f}")
        print(f"   Дисперсия: {var_loss:.25f}")
        print(f"   Минимальный loss: {min_loss:.15f}")
        print(f"   Максимальный loss: {max_loss:.15f}")
        print(f"   Размах: {max_loss - min_loss:.15f}")
        print(f"   Средняя точность: {np.mean(accuracies):.12f}")
        
        # Анализ весов
        weights = np.array([r['params'][0][0] for r in self.results])
        weights2 = np.array([r['params'][0][1] for r in self.results])
        biases = np.array([r['bias'] for r in self.results])
        
        print(f"\n📈 АНАЛИЗ ВЕСОВ:")
        print(f"   {'='*50}")
        print(f"   Вес w1: среднее={np.mean(weights):.8f}, std={np.std(weights):.8f}")
        print(f"   Вес w2: среднее={np.mean(weights2):.8f}, std={np.std(weights2):.8f}")
        print(f"   Смещение bias: среднее={np.mean(biases):.8f}, std={np.std(biases):.8f}")
        
        # Корреляция между весами и loss
        corr_w1 = np.corrcoef(weights, final_losses)[0, 1]
        corr_w2 = np.corrcoef(weights2, final_losses)[0, 1]
        print(f"\n📉 КОРРЕЛЯЦИЯ С LOSS:")
        print(f"   w1 vs loss: {corr_w1:.6f}")
        print(f"   w2 vs loss: {corr_w2:.6f}")
        
        # Статистический тест на аттрактор
        # Если дисперсия меньше 1e-10, считаем, что аттрактор есть
        if var_loss < 1e-10:
            print(f"\n{'='*60}")
            print(f"✅ АТТРАКТОР ПОДТВЕРЖДЁН")
            print(f"   Дисперсия {var_loss:.2e} < 1e-10")
            print(f"   Все {self.n_runs} запусков сошлись к одному loss")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"⚠️ АТТРАКТОР НЕ ДОСТИГНУТ (или нужна更高 точность)")
            print(f"   Дисперсия {var_loss:.2e} >= 1e-10")
            print(f"{'='*60}")
        
        # Визуализация
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Распределение loss
        axes[0, 0].hist(final_losses, bins=30, edgecolor='black', color='steelblue')
        axes[0, 0].axvline(mean_loss, color='red', linestyle='--', 
                          label=f'mean={mean_loss:.2e}')
        axes[0, 0].set_xlabel('Финальный loss')
        axes[0, 0].set_ylabel('Частота')
        axes[0, 0].set_title(f'Распределение потерь (дисперсия={var_loss:.2e})')
        axes[0, 0].legend()
        
        # 2. Кривые обучения (лог шкала)
        axes[0, 1].set_title('Кривые обучения (все запуски)')
        for r in self.results:
            axes[0, 1].semilogy(r['history'][::100], alpha=0.3, color='steelblue')
        axes[0, 1].set_xlabel('Эпоха (x100)')
        axes[0, 1].set_ylabel('Loss (лог шкала)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Последние 1000 эпох (выход на плато)
        axes[0, 2].set_title('Выход на плато (последние 1000 эпох)')
        for r in self.results:
            axes[0, 2].plot(r['history'][-1000:], alpha=0.3, color='steelblue')
        axes[0, 2].set_xlabel('Эпоха (от конца)')
        axes[0, 2].set_ylabel('Loss')
        
        # 4. Разброс весов
        axes[1, 0].scatter(range(self.n_runs), weights, alpha=0.7, label='w1', s=20)
        axes[1, 0].scatter(range(self.n_runs), weights2, alpha=0.7, label='w2', s=20)
        axes[1, 0].set_xlabel('Номер запуска')
        axes[1, 0].set_ylabel('Значение веса')
        axes[1, 0].set_title('Финальные значения весов')
        axes[1, 0].legend()
        
        # 5. Веса vs loss
        axes[1, 1].scatter(weights, final_losses, alpha=0.7, label='w1', s=30)
        axes[1, 1].scatter(weights2, final_losses, alpha=0.7, label='w2', s=30)
        axes[1, 1].set_xlabel('Вес')
        axes[1, 1].set_ylabel('Финальный loss')
        axes[1, 1].set_title('Зависимость loss от весов')
        axes[1, 1].legend()
        
        # 6. Гистограмма весов
        axes[1, 2].hist(weights, bins=30, alpha=0.5, label='w1', edgecolor='black')
        axes[1, 2].hist(weights2, bins=30, alpha=0.5, label='w2', edgecolor='black')
        axes[1, 2].set_xlabel('Значение')
        axes[1, 2].set_ylabel('Частота')
        axes[1, 2].set_title('Распределение весов')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('level0_ultimate_precision.png', dpi=150)
        print(f"\n📁 График сохранён: level0_ultimate_precision.png")
        
        # Сохраняем данные
        df = pd.DataFrame([
            {
                'seed': r['seed'],
                'final_loss': r['final_loss'],
                'accuracy': r['accuracy'],
                'w1': r['params'][0][0],
                'w2': r['params'][0][1],
                'bias': r['bias']
            } for r in self.results
        ])
        df.to_csv('level0_ultimate_results.csv', index=False)
        print(f"📁 Данные сохранены: level0_ultimate_results.csv")
        
        # Статистический отчёт
        print(f"\n📊 СТАТИСТИЧЕСКИЙ ОТЧЁТ:")
        print(f"   {'='*50}")
        print(f"   Доверительный интервал (95%): {mean_loss:.2e} ± {1.96*std_loss/np.sqrt(self.n_runs):.2e}")
        print(f"   Коэффициент вариации: {std_loss/mean_loss:.2e}")
        print(f"   Размах/среднее: {(max_loss-min_loss)/mean_loss:.2e}")
        
        return var_loss

if __name__ == "__main__":
    detector = Level0_UltimatePrecision(n_runs=100, n_epochs=5000, fine_tune_epochs=2000)
    detector.run_all()
