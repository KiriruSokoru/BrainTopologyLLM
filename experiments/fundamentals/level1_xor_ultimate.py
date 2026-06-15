# experiments/fundamentals/level1_xor_ultimate.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

class XOR_UltimatePrecision:
    """XOR задача — есть ли аттрактор у простейшей нелинейности?"""
    
    def __init__(self, n_runs=100, n_epochs=10000):
        self.n_runs = n_runs
        self.n_epochs = n_epochs
        self.results = []
        
    def generate_xor_data(self):
        """Классическая XOR таблица (идеальные данные)"""
        X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        # Увеличиваем датасет в 200 раз для стабильности градиента
        X = X.repeat(200, 1)
        y = y.repeat(200, 1)
        return X, y
    
    def create_model(self, hidden_size=2):
        """2-2-1 сеть с ReLU и Sigmoid на выходе"""
        return nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def train_model(self, seed, hidden_size=2):
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
        
        # Тонкая настройка
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(2000):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        final_loss = losses[-1]
        
        # Проверка точности
        with torch.no_grad():
            X_test = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
            pred = model(X_test).round()
            accuracy = (pred.squeeze() == torch.tensor([0,1,1,0])).float().mean().item()
        
        return {
            'seed': seed,
            'final_loss': final_loss,
            'accuracy': accuracy,
            'history': losses,
            'hidden_weights': model[0].weight.data.numpy().copy(),
            'output_weights': model[2].weight.data.numpy().copy()
        }
    
    def run_all(self):
        print("\n🔬 УРОВЕНЬ 1: XOR ЗАДАЧА (ПРЕДЕЛЬНАЯ ТОЧНОСТЬ)")
        print("="*60)
        print(f"   Запусков: {self.n_runs}")
        print(f"   Эпох: {self.n_epochs + 2000}")
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
        
        print(f"\n📊 РЕЗУЛЬТАТЫ XOR (n={self.n_runs} запусков):")
        print(f"   Средний финальный loss: {mean_loss:.10f}")
        print(f"   Стандартное отклонение: {std_loss:.10f}")
        print(f"   Дисперсия: {var_loss:.15f}")
        print(f"   Средняя точность: {np.mean(accuracies):.6f}")
        
        if var_loss < 1e-8:
            print(f"\n✅ АТТРАКТОР ЕСТЬ (дисперсия {var_loss:.2e} < 1e-8)")
        else:
            print(f"\n❌ АТТРАКТОРА НЕТ (дисперсия {var_loss:.2e} >= 1e-8)")
            print(f"   Нелинейность создаёт множество решений")
        
        # Визуализация
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(final_losses, bins=30, edgecolor='black')
        axes[0, 0].set_title(f'Распределение потерь (var={var_loss:.2e})')
        
        axes[0, 1].set_title('Кривые обучения')
        for r in self.results[:10]:
            axes[0, 1].semilogy(r['history'][::200], alpha=0.7)
        
        axes[1, 0].scatter(range(self.n_runs), accuracies, alpha=0.7)
        axes[1, 0].set_title('Точность по запускам')
        
        axes[1, 1].hist(accuracies, bins=20, edgecolor='black', color='green')
        axes[1, 1].set_title('Распределение точности')
        
        plt.tight_layout()
        plt.savefig('level1_xor_ultimate_results.png', dpi=150)
        print(f"\n📁 График: level1_xor_ultimate_results.png")
        
        # Сохраняем данные
        df = pd.DataFrame([
            {
                'seed': r['seed'],
                'final_loss': r['final_loss'],
                'accuracy': r['accuracy']
            } for r in self.results
        ])
        df.to_csv('level1_xor_results.csv', index=False)
        
        return var_loss

if __name__ == "__main__":
    detector = XOR_UltimatePrecision(n_runs=100, n_epochs=10000)
    detector.run_all()
