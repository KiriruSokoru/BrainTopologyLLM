# experiments/fundamentals/run_all_levels.py
import subprocess
import sys

levels = [
    ("level0_single_neuron.py", "Один нейрон (логистическая регрессия)"),
    ("level1_xor.py", "XOR задача (2-2-1)"),
]

for script, description in levels:
    print(f"\n{'='*60}")
    print(f"Запуск: {description}")
    print(f"{'='*60}")
    subprocess.run([sys.executable, script])
    
    input("\nНажми Enter для следующего уровня...")
