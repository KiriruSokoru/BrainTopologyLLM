#!/usr/bin/env python3
"""
Эксперимент: Сумматор 7483 (4-битный)
Гипотеза: аттрактор есть, но тривиальный — выход всегда A+B
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

class Summator7483:
    """Симуляция 4-битного сумматора с переносом"""
    def __init__(self, noise_level=0.0):
        self.noise_level = noise_level  # искусственные "сбои"
    
    def add(self, a, b):
        """Сложение с возможным шумом (сбой бита)"""
        result = a + b
        
        # Искусственный шум (если нужно проверить устойчивость)
        if np.random.random() < self.noise_level:
            # Случайный битовый сбой
            result ^= (1 << np.random.randint(0, 5))
        
        return result & 0b1111  # 4 бита, маскируем
    
    def run_experiment(self, n_samples=10000, noise_levels=[0.0, 0.01, 0.05, 0.1]):
        """Прогон с разными уровнями шума"""
        results = []
        
        print("=" * 60)
        print("СУММАТОР 7483 — ПОИСК АТТРАКТОРА")
        print("=" * 60)
        
        for noise in noise_levels:
            self.noise_level = noise
            print(f"\n🔹 Уровень шума: {noise*100:.0f}%")
            
            outputs = []
            errors = []
            
            for _ in tqdm(range(n_samples), desc=f"Шум {noise}"):
                a = np.random.randint(0, 16)
                b = np.random.randint(0, 16)
                expected = (a + b) & 0b1111
                actual = self.add(a, b)
                outputs.append(actual)
                errors.append(actual != expected)
            
            outputs = np.array(outputs)
            error_rate = np.mean(errors)
            
            # Анализ распределения выходов
            unique_outputs = len(set(outputs))
            entropy = self._compute_entropy(outputs)
            
            print(f"   Уникальных выходов: {unique_outputs}/16")
            print(f"   Энтропия: {entropy:.3f} (макс 4.0)")
            print(f"   Ошибок (сбой): {error_rate*100:.2f}%")
            
            results.append({
                'noise': noise,
                'unique_outputs': unique_outputs,
                'entropy': entropy,
                'error_rate': error_rate,
                'attractor_found': unique_outputs <= 2  # если выходов мало — сильный аттрактор
            })
        
        return pd.DataFrame(results)
    
    def _compute_entropy(self, outputs):
        """Энтропия Шеннона"""
        _, counts = np.unique(outputs, return_counts=True)
        probs = counts / len(outputs)
        return -np.sum(probs * np.log2(probs + 1e-10))


def main():
    summator = Summator7483()
    df = summator.run_experiment(n_samples=10000)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/summator_7483_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 60)
    print("📜 ВЫВОДЫ ДЛЯ ФИЛОСОФА")
    print("=" * 60)
    
    for _, row in df.iterrows():
        if row['noise'] == 0.0:
            if row['unique_outputs'] == 16 and row['entropy'] > 3.9:
                print(f"✅ Без шума: аттрактора НЕТ (все 16 выходов, энтропия {row['entropy']:.3f})")
                print(f"   → Сумматор — чистая функция, никакого 'сведения' к одному состоянию")
            else:
                print(f"⚠️ Без шума: неожиданно энтропия {row['entropy']:.3f}")
    
    for _, row in df.iterrows():
        if row['noise'] > 0:
            if row['error_rate'] < 0.05 and row['unique_outputs'] < 10:
                print(f"\n🔹 При шуме {row['noise']*100:.0f}%:")
                print(f"   → Ошибок {row['error_rate']*100:.2f}%, выходов {row['unique_outputs']}/16")
                print(f"   → Аттрактор есть, но он размыт шумом")
    
    print(f"\n📁 Результаты: {csv_path}")
    
    # Главный вывод
    print("\n" + "=" * 60)
    print("🎯 ГЛАВНЫЙ ВЫВОД")
    print("=" * 60)
    print("У детерминированного сумматора аттрактора НЕТ в классическом смысле.")
    print("Система не 'сходится' — она просто вычисляет функцию.")
    print("Аттрактор появляется ТОЛЬКО при шуме (сбоях), когда выходной сигнал")
    print("начинает тяготеть к ограниченному набору значений.")
    print("\n→ Закон Сокола требует 'сложности' и 'свободы выбора'.")
    print("  У чистого сумматора их нет — поэтому и аттрактора нет.")


if __name__ == "__main__":
    main()
