import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from factory import BrokenResNetFactory
from trainer import ResNetTrainer
from surgery import RicciSurgery
from checkpoint import CheckpointManager

def run_single_experiment(skip_strength, exp_name, device='cpu', 
                          epochs_before=5, epochs_after=5, resume=True):
    """Запускает один эксперимент с поддержкой чекпоинтов"""
    
    print(f"\n{'='*70}")
    print(f"🧪 ЭКСПЕРИМЕНТ: {exp_name}")
    print(f"   skip_strength = {skip_strength}")
    print(f"   BatchNorm = удалён")
    print(f"   Resume mode = {'ON' if resume else 'OFF'}")
    print(f"{'='*70}\n")
    
    # Инициализируем чекпоинт-менеджер
    checkpoint_mgr = CheckpointManager(exp_name, checkpoint_dir='./checkpoints')
    
    if resume:
        checkpoint_mgr.list_checkpoints()
    
    # Создаём модель
    with tqdm(total=1, desc="🏗️ Создание модели", leave=False) as pbar:
        model = BrokenResNetFactory.create(skip_strength=skip_strength, remove_batchnorm=True, pretrained=False)
        model = model.to(device)
        pbar.update(1)
    
    # Инициализируем тренера
    trainer = ResNetTrainer(model, device=device, batch_size=64, learning_rate=0.001, exp_name=exp_name)
    
    # Пытаемся восстановить из чекпоинта
    start_epoch = 0
    start_phase = 'start'
    surgery_log = None
    surgery_performed = False
    
    if resume:
        loaded = checkpoint_mgr.load(model, trainer.optimizer)
        if loaded is not None and len(loaded) == 5:
            metadata, history, surgery_log, model, optimizer = loaded
            trainer.optimizer = optimizer
            if metadata:
                start_epoch = metadata.get('epoch', 0)
                start_phase = metadata.get('phase', 'start')
                if history:
                    trainer.history = history
                if surgery_log:
                    surgery_performed = True
                print(f"\n  🔄 Возобновляем с эпохи {start_epoch}, фаза '{start_phase}'")
        else:
            print("  📭 Чекпоинтов не найдено, начинаем с нуля")
    
    # Фаза 1: Обучение до хирургии
    if not surgery_performed and start_phase != 'after_surgery':
        print("\n📚 ФАЗА 1: ОБУЧЕНИЕ ДО ХИРУРГИИ")
        print("-" * 50)
        
        for epoch in range(start_epoch, epochs_before):
            print(f"\n  Эпоха {epoch+1}/{epochs_before}")
            
            # Тренировка
            loss, _ = trainer.train_epoch(epoch_num=epoch+1)
            
            # Валидация
            val_acc = trainer.evaluate(desc=f"  Валидация {epoch+1}")
            
            # Диагностика
            diag = trainer.diagnose_singularities(num_batches=3, desc=f"  Диагностика {epoch+1}")
            
            # Обновляем историю
            trainer.update_history(epoch+1, loss, val_acc, diag)
            
            print(f"    📈 Loss={loss:.4f}, Acc={val_acc:.2f}%, "
                  f"Мастеров={diag['masters']:.0f}, Сингулярностей={diag['singularities']:.0f}")
            
            # Сохраняем чекпоинт после каждой эпохи
            checkpoint_mgr.save(
                model, trainer.optimizer, trainer.history, 
                surgery_log=None, epoch=epoch+1, phase='before_surgery'
            )
        
        # Фаза 2: Хирургия (только если есть сингулярности)
        print(f"\n🔪 ФАЗА 2: ХИРУРГИЯ ПЕРЕЛЬМАНА")
        print("-" * 50)
        
        # Проверяем, есть ли сингулярности
        diag_before = trainer.diagnose_singularities(num_batches=3)
        if diag_before['singularities'] > 0:
            surgeon = RicciSurgery(model, device=device)
            pre_surgery_stats, surgery_log = surgeon.perform_surgery(trainer.val_loader, verbose=True)
            surgery_performed = True
        else:
            print("  ✅ Сингулярностей нет, хирургия не требуется")
            surgery_performed = True  # Помечаем как выполненную, чтобы перейти к фазе 3
        
        # Сохраняем чекпоинт после хирургии
        checkpoint_mgr.save(
            model, trainer.optimizer, trainer.history, 
            surgery_log=surgery_log, epoch=epochs_before, phase='after_surgery'
        )
    
    # Фаза 3: Дообучение после хирургии
    print(f"\n📚 ФАЗА 3: ДООБУЧЕНИЕ ПОСЛЕ ХИРУРГИИ")
    print("-" * 50)
    
    # Определяем, сколько эпох уже сделано после хирургии
    current_total_epochs = len(trainer.history['epoch'])
    epochs_done_after = max(0, current_total_epochs - epochs_before)
    
    for epoch_offset in range(epochs_done_after, epochs_after):
        current_epoch_num = epochs_before + epoch_offset + 1
        print(f"\n  Эпоха {current_epoch_num}/{epochs_before+epochs_after}")
        
        # Тренировка
        loss, _ = trainer.train_epoch(epoch_num=current_epoch_num)
        
        # Валидация
        val_acc = trainer.evaluate(desc=f"  Валидация {current_epoch_num}")
        
        # Диагностика
        diag = trainer.diagnose_singularities(num_batches=3, desc=f"  Диагностика {current_epoch_num}")
        
        # Обновляем историю
        trainer.update_history(current_epoch_num, loss, val_acc, diag)
        
        print(f"    📈 Loss={loss:.4f}, Acc={val_acc:.2f}%, "
              f"Мастеров={diag['masters']:.0f}, Сингулярностей={diag['singularities']:.0f}")
        
        # Сохраняем чекпоинт
        checkpoint_mgr.save(
            model, trainer.optimizer, trainer.history, 
            surgery_log=surgery_log, epoch=current_epoch_num, phase='after_surgery'
        )
    
    # Сохраняем финальные результаты
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'exp_name': exp_name,
        'skip_strength': skip_strength,
        'final_accuracy': trainer.history['accuracy'][-1] if trainer.history['accuracy'] else 0,
        'peak_accuracy': max(trainer.history['accuracy']) if trainer.history['accuracy'] else 0,
        'accuracy_before_surgery': trainer.history['accuracy'][epochs_before - 1] if len(trainer.history['accuracy']) >= epochs_before else 0,
        'accuracy_after_surgery': trainer.history['accuracy'][-1] if trainer.history['accuracy'] else 0,
        'singularities_peak': max(trainer.history['singularities']) if trainer.history['singularities'] else 0,
        'singularities_before_surgery': trainer.history['singularities'][epochs_before - 1] if len(trainer.history['singularities']) >= epochs_before else 0,
        'singularities_after_surgery': trainer.history['singularities'][-1] if trainer.history['singularities'] else 0,
        'surgery_log': surgery_log,
        'history': trainer.history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(results_dir, f'{exp_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_results(trainer.history, exp_name, skip_strength, results_dir)
    
    # Финальный чекпоинт
    checkpoint_mgr.save(
        model, trainer.optimizer, trainer.history, 
        surgery_log=surgery_log, epoch=epochs_before+epochs_after, phase='completed'
    )
    
    print(f"\n  ✅ Эксперимент {exp_name} завершён!")
    
    return results

def plot_results(history, exp_name, skip_strength, results_dir):
    """Строит графики"""
    if not history['epoch']:
        print(f"  ⚠️ Нет данных для построения графиков {exp_name}")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['epoch'], history['accuracy'], 'b-', linewidth=2)
    if len(history['epoch']) > 5:
        axes[0, 0].axvline(x=5.5, color='r', linestyle='--', linewidth=2, label='Хирургия')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title(f'{exp_name}: Точность (skip={skip_strength})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['epoch'], history['singularities'], 'r-', linewidth=2, label='Сингулярности')
    axes[0, 1].plot(history['epoch'], history['masters'], 'g-', linewidth=2, label='Мастера')
    if len(history['epoch']) > 5:
        axes[0, 1].axvline(x=5.5, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Количество')
    axes[0, 1].set_title('Динамика сингулярностей и мастеров')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['epoch'], history['loss'], 'purple', linewidth=2)
    if len(history['epoch']) > 5:
        axes[1, 0].axvline(x=5.5, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Эпоха')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Функция потерь')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Баровый график эффекта (если достаточно данных)
    if len(history['epoch']) >= 5:
        before_idx = 4
        after_idx = -1
        axes[1, 1].bar(['До хирургии', 'После хирургии'], 
                       [history['singularities'][before_idx], history['singularities'][after_idx]], 
                       color=['red', 'darkred'], alpha=0.7, label='Сингулярности')
        axes[1, 1].bar(['До хирургии', 'После хирургии'], 
                       [history['masters'][before_idx], history['masters'][after_idx]], 
                       bottom=[history['singularities'][before_idx], history['singularities'][after_idx]],
                       color=['green', 'darkgreen'], alpha=0.7, label='Мастера')
        axes[1, 1].set_ylabel('Количество')
        axes[1, 1].set_title('Эффект хирургии')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{exp_name}_plots.png'), dpi=150)
    plt.close()

def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Запуск экспериментов BrokenResNet')
    parser.add_argument('--resume', action='store_true', default=True, help='Возобновить с чекпоинта')
    parser.add_argument('--clean', action='store_true', help='Очистить все чекпоинты перед запуском')
    parser.add_argument('--exp', type=str, choices=['A', 'B', 'C', 'all'], default='all', 
                       help='Какой эксперимент запустить')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Используем устройство: {device}")
    
    # Очистка чекпоинтов если нужно
    if args.clean:
        print("🧹 Очищаем все чекпоинты...")
        for exp in ['A_legkaya_travma_skip07', 'B_sredniy_perelom_skip05', 'C_klinicheskaya_smert_skip03']:
            mgr = CheckpointManager(exp)
            mgr.clear()
    
    experiments = {
        'A': (0.7, "A_legkaya_travma_skip07"),
        'B': (0.5, "B_sredniy_perelom_skip05"),
        'C': (0.3, "C_klinicheskaya_smert_skip03")
    }
    
    if args.exp == 'all':
        to_run = experiments.items()
    else:
        to_run = [(args.exp, experiments[args.exp])]
    
    results_list = []
    
    outer_pbar = tqdm(to_run, desc="🏥 Общий прогресс экспериментов", unit='exp')
    
    for exp_key, (skip_strength, exp_name) in outer_pbar:
        outer_pbar.set_postfix({'текущий': exp_name})
        
        try:
            results = run_single_experiment(
                skip_strength=skip_strength,
                exp_name=exp_name,
                device=device,
                epochs_before=5,
                epochs_after=5,
                resume=args.resume
            )
            results_list.append(results)
            outer_pbar.set_postfix({'текущий': exp_name, '✅': 'готов'})
        except KeyboardInterrupt:
            print(f"\n\n⚠️ Эксперимент {exp_name} прерван пользователем")
            print(f"   Чекпоинт сохранён, можно возобновить")
            raise
        except Exception as e:
            print(f"\n❌ Ошибка в {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            outer_pbar.set_postfix({'текущий': exp_name, '❌': 'ошибка'})
            continue
    
    # Финальное сравнение
    if results_list:
        print("\n" + "="*70)
        print("📊 ИТОГОВОЕ СРАВНЕНИЕ")
        print("="*70)
        print(f"\n{'Эксперимент':<35} {'До хирургии':>15} {'После хирургии':>15} {'Эффект':>15}")
        print("-" * 80)
        for res in results_list:
            impr = res['accuracy_after_surgery'] - res['accuracy_before_surgery']
            print(f"{res['exp_name']:<35} {res['accuracy_before_surgery']:>14.2f}% {res['accuracy_after_surgery']:>14.2f}% {impr:>14.2f} п.п.")
    
    print("\n✅ ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")

if __name__ == "__main__":
    main()
