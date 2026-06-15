# experiments/fundamentals/level1_xor.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Level1_XOR_Attractor:
    """XOR задача — минимальная нелинейность"""
    
    def __init__(self, n_runs=30, n_epochs=1000):
        self.n_runs = n_runs
        self.n_epochs = n_epochs
        self.results = []
        
    def generate_xor_data(self):
        """Классическая XOR таблица"""
        X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        # Увеличиваем датасет повторением для стабильности
        X = X.repeat(50, 1)
        y = y.repeat(50, 1)
        return X, y
    
    def create_model(self, hidden_size=2):
        """2-2-1 сеть с ReLU"""
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
        
        # Проверка точности
        with torch.no_grad():
            X_test = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
            pred = model(X_test).round()
            accuracy = (pred.squeeze() == torch.tensor([0,1,1,0])).float().mean().item()
        
        return losses[-1], losses, accuracy
    
    def run_all(self):
        print("\n🔬 УРОВЕНЬ 1: XOR ЗАДАЧА (2-2-1)")
        print("="*50)
        
        for seed in tqdm(range(self.n_runs), desc="Запуски"):
            final_loss, history, acc = self.train_model(seed)
            self.results.append({
                'seed': seed,
                'final_loss': final_loss,
                'accuracy': acc,
                'history': history
            })
        
        self.analyze()
    
    def analyze(self):
        final_losses = [r['final_loss'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        
        mean_loss = np.mean(final_losses)
        std_loss = np.std(final_losses)
        var_loss = np.var(final_losses)
        mean_acc = np.mean(accuracies)
        
        print(f"\n📊 Результаты (n={self.n_runs} запусков):")
        print(f"   Средний финальный loss: {mean_loss:.4f}")
        print(f"   Средняя точность: {mean_acc:.3f}")
        print(f"   Дисперсия loss: {var_loss:.6f}")
        
        if var_loss < 0.01:
            print(f"\n✅ АТТРАКТОР ЕСТЬ (дисперсия {var_loss:.6f} < 0.01)")
        else:
            print(f"\n❌ АТТРАКТОРА НЕТ (дисперсия {var_loss:.6f} >= 0.01)")
        
        # Визуализация
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].hist(final_losses, bins=15, edgecolor='black')
        axes[0].set_xlabel('Финальный loss')
        axes[0].set_ylabel('Частота')
        axes[0].set_title(f'Распределение потерь (var={var_loss:.6f})')
        
        axes[1].hist(accuracies, bins=15, edgecolor='black', color='green')
        axes[1].set_xlabel('Точность')
        axes[1].set_title(f'Распределение точности (mean={mean_acc:.3f})')
        
        axes[2].set_title('Кривые обучения (5 запусков)')
        for r in self.results[:5]:
            axes[2].plot(r['history'], alpha=0.7)
        axes[2].set_xlabel('Эпоха')
        axes[2].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('level1_xor_results.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    detector = Level1_XOR_Attractor(n_runs=30, n_epochs=1000)
    detector.run_all()
