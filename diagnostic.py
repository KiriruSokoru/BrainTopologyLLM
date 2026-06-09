# diagnostic.py
import os
import sys

def diagnose_structure():
    """
    Диагностика текущей структуры BrainTopologyLLM
    Ничего не создаёт, только отчитывается
    """
    
    print("=" * 60)
    print("ДИАГНОСТИКА СТРУКТУРЫ BrainTopologyLLM")
    print("=" * 60)
    
    # Ключевые папки и файлы, которые должны быть
    expected = {
        'core_dirs': ['checkpoints', 'logs', 'config', 'data', 'models', 'surgery', 'train', 'experiments'],
        'core_files': ['README.md', 'requirements.txt', '.gitignore'],
        'existing_code': [
            'run.py', 'train.py', 'gpt2_surgery.py', 'ricci_attention.py',
            'perelman_surgery.py', 'copy_task.py', 'trainer.py'
        ]
    }
    
    # 1. Проверяем корневые папки
    print("\n📁 СУЩЕСТВУЮЩИЕ ПАПКИ:")
    existing_dirs = []
    missing_dirs = []
    
    for d in expected['core_dirs']:
        if os.path.exists(d) and os.path.isdir(d):
            existing_dirs.append(d)
            # Смотрим содержимое
            contents = os.listdir(d) if os.path.exists(d) else []
            print(f"  ✅ {d}/ ({len(contents)} файлов)")
            if contents:
                for f in contents[:3]:  # показываем первые 3
                    print(f"      - {f}")
                if len(contents) > 3:
                    print(f"      ... и {len(contents)-3} ещё")
        else:
            missing_dirs.append(d)
            print(f"  ❌ {d}/ (отсутствует)")
    
    # 2. Проверяем корневые файлы
    print("\n📄 СУЩЕСТВУЮЩИЕ ФАЙЛЫ В КОРНЕ:")
    root_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in root_files:
        if f.endswith('.py') or f.endswith('.md') or f.endswith('.txt') or f.endswith('.yaml'):
            size = os.path.getsize(f)
            print(f"  📄 {f} ({size} bytes)")
    
    # 3. Ищем существующий код, который мы уже написали
    print("\n🔍 ПОИСК СУЩЕСТВУЮЩИХ МОДУЛЕЙ:")
    
    modules_found = []
    modules_missing = []
    
    for module in expected['existing_code']:
        found = False
        for root, dirs, files in os.walk('.'):
            if module in files:
                found = True
                modules_found.append(module)
                path = os.path.join(root, module)
                print(f"  ✅ {module} -> {path}")
                break
        if not found:
            modules_missing.append(module)
            print(f"  ❌ {module} (не найден)")
    
    # 4. Проверяем, есть ли чекпоинты (признак что эксперименты уже шли)
    print("\n💾 ПРОВЕРКА ЧЕКПОЙНТОВ:")
    if os.path.exists('checkpoints'):
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt') or f.endswith('.pth')]
        if checkpoints:
            print(f"  🎯 Найдено {len(checkpoints)} чекпоинтов:")
            for ch in checkpoints[:5]:
                size = os.path.getsize(f'checkpoints/{ch}')
                print(f"      - {ch} ({size/1024:.1f} KB)")
        else:
            print("  ⚪ Чекпоинтов нет (чистый старт)")
    else:
        print("  ⚪ Папка checkpoints отсутствует")
    
    # 5. Проверяем конфиги
    print("\n⚙️ КОНФИГУРАЦИИ:")
    if os.path.exists('config'):
        configs = [f for f in os.listdir('config') if f.endswith('.yaml') or f.endswith('.yml')]
        if configs:
            for cfg in configs:
                print(f"  📋 {cfg}")
        else:
            print("  ⚪ Нет .yaml файлов")
    else:
        print("  ⚪ Папка config отсутствует")
    
    # 6. Смотрим историю git (если есть)
    print("\n📊 СТАТУС GIT (если инициализирован):")
    if os.path.exists('.git'):
        print("  ✅ Git репозиторий существует")
        try:
            import subprocess
            last_commit = subprocess.check_output(['git', 'log', '-1', '--oneline']).decode().strip()
            print(f"  📝 Последний коммит: {last_commit}")
        except:
            pass
    else:
        print("  ⚪ Git не инициализирован")
    
    # 7. ИТОГ: что нужно сделать
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ:")
    print("=" * 60)
    
    if missing_dirs:
        print(f"📁 Создать папки: {', '.join(missing_dirs)}")
    
    if modules_missing:
        print(f"📄 Создать файлы: {', '.join(modules_missing)}")
    
    if not os.path.exists('requirements.txt'):
        print("📦 Создать requirements.txt")
    
    if not os.path.exists('config/experiment_config.yaml'):
        print("⚙️ Создать config/experiment_config.yaml")
    
    if existing_dirs and not modules_missing:
        print("\n✅ Структура уже есть! Можно запускать эксперимент.")
        print("🚀 Запусти: python experiments/run_attractor_test.py")
    else:
        print("\n⚠️ Требуется донастройка структуры.")
        print("🛠️ Запусти: python setup_structure.py (БЕЗОПАСНАЯ ВЕРСИЯ)")
    
    return {
        'existing_dirs': existing_dirs,
        'missing_dirs': missing_dirs,
        'modules_found': modules_found,
        'modules_missing': modules_missing
    }

def safe_setup_structure():
    """
    Безопасное создание структуры — только то, чего нет
    """
    print("\n" + "=" * 60)
    print("БЕЗОПАСНОЕ СОЗДАНИЕ СТРУКТУРЫ")
    print("=" * 60)
    
    dirs_to_create = ['checkpoints', 'logs', 'config', 'data', 'models', 'surgery', 'train', 'experiments']
    
    created = []
    existed = []
    
    for d in dirs_to_create:
        if not os.path.exists(d):
            os.makedirs(d)
            created.append(d)
            print(f"  📁 Создана папка: {d}/")
            # Создаём __init__.py
            init_path = os.path.join(d, '__init__.py')
            with open(init_path, 'w') as f:
                f.write(f"# {d} module for BrainTopologyLLM\n")
            print(f"      └── {d}/__init__.py")
        else:
            existed.append(d)
            print(f"  ✅ Уже существует: {d}/")
    
    # Создаём requirements.txt если нет
    if not os.path.exists('requirements.txt'):
        with open('requirements.txt', 'w') as f:
            f.write("""torch>=2.0.0
transformers>=4.30.0
tqdm
numpy
tensorboard
pyyaml
matplotlib
seaborn
""")
        print("  📦 Создан requirements.txt")
    else:
        print("  ✅ requirements.txt уже есть")
    
    # Создаём .gitignore если нет
    if not os.path.exists('.gitignore'):
        with open('.gitignore', 'w') as f:
            f.write("""__pycache__/
*.pyc
checkpoints/*.pt
logs/
*.log
.DS_Store
""")
        print("  📄 Создан .gitignore")
    
    print("\n" + "=" * 60)
    print(f"ИТОГ: создано {len(created)} папок, {len(existed)} уже были")
    print("=" * 60)
    
    return created, existed

if __name__ == "__main__":
    # Сначала диагностика
    diagnosis = diagnose_structure()
    
    # Спрашиваем пользователя
    if diagnosis['missing_dirs'] or diagnosis['modules_missing']:
        print("\n❓ Желаешь создать недостающие папки и файлы? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            safe_setup_structure()
            print("\n✅ Структура подготовлена. Теперь можно копировать остальные файлы.")
        else:
            print("\n⏸️ Остановлено по твоей команде. Жду указаний.")
    else:
        print("\n✅ Всё уже есть! Продолжаем с существующей структурой.")
