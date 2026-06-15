# experiments/fundamentals/level0_single_neuron_precision.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from tqdm import tqdm
import pandas as pd

class Level0_HighPrecision:
    """Один нейрон — измеряем истинный loss с высокой точностью"""
    
    def __init__(self, n_runs=50, n_epochs=2000):
        self.n_runs = n_runs
        self.n_epochs = n_epochs
        self.results = []
        
    def generate_data(self, n_samples=1000, noise=0.2):
        """Два хорошо разделимых облака"""
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, 
                          cluster_std=noise, random_state=42)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).reshape(-1, 1)
        return X, y
    
    def train_model(self, seed):
        """Обучаем с очень маленьким learning rate для точной сходимости"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X, y = self.generate_data()
        model = nn.Linear(2, 1)
        # Очень маленький lr и большая эпоха для точной сходимости
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.BCEWithLogitsLoss()
        
        losses = []
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Финализируем с ещё меньшим lr для добивки
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        for epoch in range(500):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        final_loss = losses[-1]
        
        # Вычисляем точность классификации
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
        print("🔬 УРОВЕНЬ 0: ОДИН НЕЙРОН (ВЫСОКАЯ ТОЧНОСТЬ)")
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
        
        print(f"\n📊 Результаты (n={self.n_runs} запусков):")
        print(f"   Средний финальный loss: {mean_loss:.12f}")
        print(f"   Стандартное отклонение: {std_loss:.12f}")
        print(f"   Дисперсия: {var_loss:.20f}")
        print(f"   Минимальный loss: {min_loss:.12f}")
        print(f"   Максимальный loss: {max_loss:.12f}")
        print(f"   Средняя точность: {np.mean(accuracies):.8f}")
        
        # Анализ весов
        weights = np.array([r['params'][0][0] for r in self.results])
        biases = np.array([r['bias'] for r in self.results])
        
        print(f"\n📈 Анализ весов:")
        print(f"   Вес w1: среднее={np.mean(weights):.6f}, std={np.std(weights):.6f}")
        print(f"   Смещение bias: среднее={np.mean(biases):.6f}, std={np.std(biases):.6f}")
        
        # Проверка на точный ноль
        zero_loss_count = sum(1 for l in final_losses if l < 1e-10)
        print(f"\n🎯 Запусков с loss < 1e-10: {zero_loss_count}/{self.n_runs} ({100*zero_loss_count/self.n_runs:.1f}%)")
        
        # Критерий аттрактора
        if var_loss < 1e-6:
            print(f"\n✅ АТТРАКТОР ЕСТЬ (дисперсия {var_loss:.2e} < 1e-6)")
        else:
            print(f"\n❌ АТТРАКТОРА НЕТ (дисперсия {var_loss:.2e} >= 1e-6)")
        
        # Детальная визуализация
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Распределение финального loss
        axes[0, 0].hist(final_losses, bins=30, edgecolor='black', log=True)
        axes[0, 0].axvline(mean_loss, color='red', linestyle='--', label=f'mean={mean_loss:.2e}')
        axes[0, 0].set_xlabel('Финальный loss (логарифмическая шкала)')
        axes[0, 0].set_ylabel('Частота')
        axes[0, 0].set_title(f'Распределение потерь (var={var_loss:.2e})')
        axes[0, 0].legend()
        
        # 2. Кривые обучения (первые 10 запусков)
        axes[0, 1].set_title('Кривые обучения (первые 10 запусков, лог шкала)')
        for r in self.results[:10]:
            axes[0, 1].semilogy(r['history'][::50], alpha=0.7)  # каждый 50-й шаг для читаемости
        axes[0, 1].set_xlabel('Эпоха (x50)')
        axes[0, 1].set_ylabel('Loss')
        
        # 3. Разброс финальных весов
        axes[1, 0].scatter(range(self.n_runs), weights, alpha=0.7, label='Вес w1')
        axes[1, 0].scatter(range(self.n_runs), biases, alpha=0.7, label='Смещение bias')
        axes[1, 0].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].set_xlabel('Номер запуска')
        axes[1, 0].set_ylabel('Значение')
        axes[1, 0].set_title('Финальные значения весов и смещения')
        axes[1, 0].legend()
        
        # 4. Корреляция loss и весов
        axes[1, 1].scatter(weights, final_losses, alpha=0.7)
        axes[1, 1].set_xlabel('Вес w1')
        axes[1, 1].set_ylabel('Финальный loss')
        axes[1, 1].set_title('Зависимость loss от веса')
        
        plt.tight_layout()
        plt.savefig('level0_single_neuron_high_precision.png', dpi=150)
        print(f"\n📁 График сохранён: level0_single_neuron_high_precision.png")
        plt.show()
        
        # Сохраняем сырые данные
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
        df.to_csv('level0_results_precision.csv', index=False)
        print(f"📁 Данные сохранены: level0_results_precision.csv")
        
        return var_loss

if __name__ == "__main__":
    detector = Level0_HighPrecision(n_runs=50, n_epochs=2000)
    detector.run_all()
