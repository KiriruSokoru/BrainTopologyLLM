# experiments/fundamentals/level1_xor_threshold.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

class XOR_ThresholdFinder:
    """Находим минимальный размер сети, при котором появляется аттрактор"""
    
    def __init__(self, n_runs=50, n_epochs=5000):
        self.n_runs = n_runs
        self.n_epochs = n_epochs
        self.results = {}
        
    def generate_xor_data(self):
        """Классическая XOR"""
        X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        X = X.repeat(100, 1)
        y = y.repeat(100, 1)
        return X, y
    
    def create_model(self, hidden_size):
        """Сеть 2-hidden-1"""
        return nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def train_model(self, hidden_size, seed):
        """Обучаем одну сеть"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        X, y = self.generate_xor_data()
        model = self.create_model(hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()
        
        losses = []
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        final_loss = losses[-1]
        
        with torch.no_grad():
            X_test = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
            pred = model(X_test).round()
            accuracy = (pred.squeeze() == torch.tensor([0,1,1,0])).float().mean().item()
        
        return final_loss, accuracy
    
    def test_hidden_size(self, hidden_size):
        """Тестируем один размер скрытого слоя"""
        print(f"\n📊 Тестируем hidden_size = {hidden_size}")
        losses = []
        accuracies = []
        
        for seed in tqdm(range(self.n_runs), desc=f"  Запуски", leave=False):
            loss, acc = self.train_model(hidden_size, seed)
            losses.append(loss)
            accuracies.append(acc)
        
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        var_loss = np.var(losses)
        mean_acc = np.mean(accuracies)
        
        # Критерий аттрактора: дисперсия loss < 1e-4
        has_attractor = var_loss < 1e-4
        
        print(f"    mean_loss={mean_loss:.4f}, var_loss={var_loss:.2e}, mean_acc={mean_acc:.3f}")
        print(f"    {'✅ АТТРАКТОР' if has_attractor else '❌ НЕТ АТТРАКТОРА'}")
        
        return {
            'hidden_size': hidden_size,
            'mean_loss': mean_loss,
            'std_loss': std_loss,
            'var_loss': var_loss,
            'mean_acc': mean_acc,
            'has_attractor': has_attractor,
            'losses': losses,
            'accuracies': accuracies
        }
    
    def run_all(self, hidden_sizes):
        """Запускаем для всех размеров"""
        print("🔬 XOR: ПОИСК КРИТИЧЕСКОГО РАЗМЕРА СЕТИ")
        print("="*60)
        print(f"   Запусков на размер: {self.n_runs}")
        print(f"   Эпох на запуск: {self.n_epochs}")
        print(f"   Тестируем: {hidden_sizes}")
        print("="*60)
        
        for hidden in hidden_sizes:
            result = self.test_hidden_size(hidden)
            self.results[hidden] = result
        
        self.analyze()
        return self.results
    
    def analyze(self):
        """Анализируем порог появления аттрактора"""
        
        print("\n" + "="*60)
        print("📊 АНАЛИЗ ПОРОГА АТТРАКТОРА")
        print("="*60)
        
        # Находим первый размер с аттрактором
        threshold = None
        for hidden, res in sorted(self.results.items()):
            if res['has_attractor']:
                threshold = hidden
                break
        
        if threshold:
            print(f"\n✅ КРИТИЧЕСКИЙ РАЗМЕР: hidden_size = {threshold}")
            print(f"   При этом размере дисперсия = {self.results[threshold]['var_loss']:.2e} < 1e-4")
            print(f"   Точность = {self.results[threshold]['mean_acc']:.3f}")
        else:
            print(f"\n❌ АТТРАКТОР НЕ НАЙДЕН для hidden_size ≤ {max(self.results.keys())}")
            print(f"   Возможно, нужна ещё б\'ольшая сеть или больше эпох")
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        sizes = sorted(self.results.keys())
        variances = [self.results[s]['var_loss'] for s in sizes]
        accuracies = [self.results[s]['mean_acc'] for s in sizes]
        
        # График дисперсии
        axes[0, 0].semilogy(sizes, variances, 'o-', color='red', linewidth=2, markersize=8)
        axes[0, 0].axhline(1e-4, color='green', linestyle='--', label='Порог 1e-4')
        axes[0, 0].set_xlabel('Размер скрытого слоя')
        axes[0, 0].set_ylabel('Дисперсия loss (лог шкала)')
        axes[0, 0].set_title('Зависимость дисперсии от размера сети')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # График точности
        axes[0, 1].plot(sizes, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
        axes[0, 1].axhline(1.0, color='green', linestyle='--', label='Идеал')
        axes[0, 1].set_xlabel('Размер скрытого слоя')
        axes[0, 1].set_ylabel('Средняя точность')
        axes[0, 1].set_title('Зависимость точности от размера сети')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Гистограммы для малой и большой сети
        small_size = min(sizes)
        large_size = max(sizes) if threshold else sizes[-1]
        
        small_losses = self.results[small_size]['losses']
        axes[1, 0].hist(small_losses, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title(f'Распределение loss (hidden={small_size})')
        axes[1, 0].set_xlabel('Loss')
        
        large_losses = self.results[large_size]['losses']
        axes[1, 1].hist(large_losses, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1, 1].set_title(f'Распределение loss (hidden={large_size})')
        axes[1, 1].set_xlabel('Loss')
        
        plt.tight_layout()
        plt.savefig('level1_xor_threshold_results.png', dpi=150)
        print(f"\n📁 График сохранён: level1_xor_threshold_results.png")
        
        # Сохраняем данные
        df = pd.DataFrame([
            {
                'hidden_size': s,
                'mean_loss': self.results[s]['mean_loss'],
                'var_loss': self.results[s]['var_loss'],
                'mean_acc': self.results[s]['mean_acc'],
                'has_attractor': self.results[s]['has_attractor']
            } for s in sizes
        ])
        df.to_csv('xor_threshold_results.csv', index=False)
        print(f"📁 Данные сохранены: xor_threshold_results.csv")
        
        return threshold

if __name__ == "__main__":
    # Тестируем от 2 до 64 нейронов в скрытом слое
    hidden_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    
    finder = XOR_ThresholdFinder(n_runs=40, n_epochs=5000)
    threshold = finder.run_all(hidden_sizes)
    
    print("\n" + "="*60)
    if threshold:
        print(f"🎯 ИТОГ: Аттрактор появляется при hidden_size = {threshold}")
        print(f"   Это минимальный размер сети, способной к самоорганизации")
    else:
        print(f"🎯 ИТОГ: Аттрактор не найден в диапазоне 2-64")
        print(f"   Возможно, нужна сеть больше 64 нейронов")
    print("="*60)
