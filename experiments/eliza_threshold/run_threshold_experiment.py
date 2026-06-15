#!/usr/bin/env python3
"""
Эксперимент: Семантический аттрактор ELIZA
Гипотеза: hit_rate (доля осмысленных ответов) сходится к константе,
независимо от собеседника (Закон Сокола для диалога)
"""
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from attractor_analyzer import AttractorAnalyzer
from classic_eliza import ClassicELIZA
from user_models import SovietCitizen, JewishInterlocutor, GermanInterlocutor, MoodyInterlocutor
from checkpoint_manager import ELIZACheckpointManager


def generate_100_phrases():
    """Генерирует 100 начальных фраз"""
    np.random.seed(42)
    
    templates_one = [
        "I feel {}", "I am {}", "You always {}", "Why do you {}?",
        "I think {}", "I dream about {}", "I hate {}", "I love {}",
        "Tell me about {}", "Is it true that {}?", "What if {}?",
        "The meaning of {} is", "People say I'm {}", "Today I {}",
        "I want to {}", "I don't understand {}", "Can you explain {}?",
        "{} scares me", "I remember {}", "Maybe I should {}", "Why does {} happen?"
    ]
    
    templates_two = [
        "My {} is {}", "I feel {} about {}", "{} made me {}", "I think {} is {}"
    ]
    
    adjectives = ["sad", "happy", "angry", "confused", "tired", "hopeful", "lonely", 
                  "excited", "empty", "strong", "weak", "lost", "found", "guilty"]
    nouns = ["mother", "father", "friend", "boss", "dog", "life", "work", "dream", 
             "past", "future", "doctor", "teacher", "child"]
    verbs = ["failed", "succeeded", "ran away", "came back", "lied", "told truth", 
             "forgot", "remembered", "cried", "laughed"]
    abstracts = ["love", "death", "freedom", "justice", "time", "consciousness", 
                 "God", "nothing", "everything", "silence", "pain", "joy"]
    
    phrases = set()
    
    while len(phrases) < 100:
        if np.random.random() < 0.8:
            template = np.random.choice(templates_one)
            filler = np.random.choice(adjectives + nouns + verbs + abstracts)
            phrase = template.format(filler)
        else:
            template = np.random.choice(templates_two)
            filler1 = np.random.choice(adjectives + nouns + abstracts)
            filler2 = np.random.choice(adjectives + verbs + abstracts)
            phrase = template.format(filler1, filler2)
        
        if len(phrase) < 70:
            phrases.add(phrase.capitalize())
    
    return list(phrases)[:100]


def run_mode_with_checkpoint(mode_name, user_class, start_phrases, 
                              repeats=48, max_turns=100, force_restart=False):
    """Запускает один режим с чекпоинтами"""
    print(f"\n{'='*60}")
    print(f"▶ {mode_name}")
    print(f"  Собеседник: {user_class.__name__}")
    print(f"  Фраз: {len(start_phrases)} × Повторов: {repeats} = {len(start_phrases)*repeats} диалогов")
    print(f"  Макс. ходов на диалог: {max_turns}")
    print(f"{'='*60}")
    
    checkpoint_mgr = ELIZACheckpointManager(f"semantic_{mode_name.lower()}")
    
    if force_restart:
        checkpoint_mgr.clear_checkpoint()
        completed = set()
        results = []
    else:
        cp = checkpoint_mgr.load_checkpoint()
        if cp and cp.get('mode') == mode_name:
            completed = set(cp['completed_combinations'])
            results = cp['accumulated_results']
            print(f"  Продолжаем, уже обработано {len(results)} диалогов")
        else:
            completed = set()
            results = []
    
    all_combinations = []
    for phrase_idx, phrase in enumerate(start_phrases):
        for rep in range(repeats):
            combo_key = f"{phrase_idx}_{rep}"
            if combo_key not in completed:
                all_combinations.append((phrase_idx, phrase, rep))
    
    if not all_combinations:
        print("  ✅ Все комбинации уже выполнены")
        return pd.DataFrame(results)
    
    print(f"  Осталось выполнить: {len(all_combinations)} диалогов")
    
    eliza = ClassicELIZA()
    user_model = user_class()
    analyzer = AttractorAnalyzer(eliza, user_model=user_model, max_turns=max_turns)
    
    pbar = tqdm(all_combinations, desc=f"{mode_name}", unit="dialog", leave=True)
    
    for idx, (phrase_idx, phrase, rep) in enumerate(pbar):
        outcome = analyzer.run_dialogue(phrase)
        
        result = {
            'mode': mode_name,
            'user_type': user_class.__name__.lower().replace('interlocutor', ''),
            'start_phrase': phrase,
            'phrase_idx': phrase_idx,
            'repeat': rep,
            'hit_rate': outcome['hit_rate'],
            'miss_rate': outcome['miss_rate'],
            'elephant_rate': outcome['elephant_rate'],
            'total_turns': outcome['total_turns'],
            'final_classification': outcome['final_classification'],
            'hit_rate_last_10': outcome['hit_rate_last_10'],
            'miss_rate_last_10': outcome['miss_rate_last_10'],
            'elephant_rate_last_10': outcome['elephant_rate_last_10']
        }
        results.append(result)
        
        if (idx + 1) % 50 == 0:
            completed_set = [f"{r['phrase_idx']}_{r['repeat']}" for r in results]
            checkpoint_mgr.save_checkpoint(completed_set, idx + 1, results, mode_name)
            pbar.set_postfix({"hit": f"{np.mean([r['hit_rate'] for r in results[-50:]]):.3f}"})
    
    checkpoint_mgr.clear_checkpoint()
    
    avg_hit = np.mean([r['hit_rate'] for r in results])
    print(f"  ✅ {mode_name} завершён: {len(results)} диалогов, средний hit_rate = {avg_hit:.3f}")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Семантический аттрактор ELIZA')
    parser.add_argument('--force_restart', action='store_true')
    parser.add_argument('--mode', choices=['soviet', 'jewish', 'german', 'all'], default='all')
    parser.add_argument('--test', action='store_true', help='Тестовый режим: 5 фраз, 2 повтора')
    parser.add_argument('--recognition', action='store_true',
                        help='Режим узнавания: 20 циклов с переключением стратегий')

    args = parser.parse_args()

    print("=" * 70)
    print("🧠 СЕМАНТИЧЕСКИЙ АТТРАКТОР ELIZA")
    print("   Гипотеза: hit_rate сходится к константе (Закон Сокола)")
    print("=" * 70)
    
    if args.test:
        start_phrases = generate_100_phrases()[:5]
        repeats = 2
        print(f"\n🧪 ТЕСТОВЫЙ РЕЖИМ: {len(start_phrases)} фраз, {repeats} повторов")
    else:
        start_phrases = generate_100_phrases()
        repeats = 48
        print(f"\n📝 Сгенерировано {len(start_phrases)} начальных фраз")
    
    MAX_TURNS = 100
    results_dir = 'experiments/eliza_threshold/results'
    os.makedirs(results_dir, exist_ok=True)

    if args.recognition:
        print("\n🧠 РЕЖИМ УЗНАВАНИЯ: ELIZA vs MoodyInterlocutor (20 циклов, смена стратегий)")

        eliza = ClassicELIZA()
        user = MoodyInterlocutor(seed=42, switch_every=len(start_phrases) * repeats)
        analyzer = AttractorAnalyzer(eliza, user, max_turns=MAX_TURNS)

        all_hit_rates = []
        cycle_hit_rates = []

        for cycle in range(20):
            print(f"\n--- Цикл {cycle+1}/20 ---")
            df_cycle = analyzer.run_experiment(start_phrases, repeats_per_phrase=repeats, verbose=True)
            mean_hit = df_cycle['hit_rate'].mean()
            all_hit_rates.append(mean_hit)
            cycle_hit_rates.append({
                'cycle': cycle,
                'strategy': user.schedule[cycle * user.switch_every],
                'hit_rate': mean_hit
            })

            df_cycle.to_csv(os.path.join(results_dir, f'cycle_{cycle+1}.csv'), index=False)

        print("\n" + "=" * 70)
        print("📊 АНАЛИЗ УЗНАВАНИЯ")
        print("=" * 70)

        for i in range(1, len(cycle_hit_rates)):
            prev = cycle_hit_rates[i-1]
            curr = cycle_hit_rates[i]

            if prev['strategy'] != curr['strategy']:
                expected = {0: 0.941, 1: 0.286, 2: 0.282}[curr['strategy']]
                ratio = curr['hit_rate'] / expected

                print(f"\nЦикл {i}: смена {prev['strategy']}→{curr['strategy']}")
                print(f"  hit_rate = {curr['hit_rate']:.3f} (ожидалось {expected:.3f})")
                print(f"  Коэффициент узнавания = {ratio:.2f}")

                if 0.9 < ratio < 1.1:
                    print("  ✅ ELIZa УЗНАЛА СОБЕСЕДНИКА! hit_rate мгновенно стал как у чистого режима")
                else:
                    print("  ❌ ELIZA НЕ УЗНАЛА, подстраивается заново")
        return

    modes_config = {
        'soviet': ("SOVIET", SovietCitizen),
        'jewish': ("JEWISH", JewishInterlocutor),
        'german': ("GERMAN", GermanInterlocutor)
    }
    
    all_results = []
    modes_to_run = [args.mode] if args.mode != 'all' else list(modes_config.keys())
    
    for mode_key in modes_to_run:
        mode_name, user_class = modes_config[mode_key]
        df = run_mode_with_checkpoint(mode_name, user_class, start_phrases, 
                                       repeats=repeats, max_turns=MAX_TURNS, 
                                       force_restart=args.force_restart)
        all_results.append(df)
    
    if not all_results:
        print("❌ Нет данных")
        return
    
    final_df = pd.concat(all_results, ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f'semantic_attractor_{timestamp}.csv')
    final_df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("📊 СЕМАНТИЧЕСКИЙ АТТРАКТОР: hit_rate (доля осмысленных ответов)")
    print("   Если Закон Сокола работает — hit_rate одинаков у всех собеседников")
    print("=" * 70)
    
    summary = final_df.groupby('mode').agg({
        'hit_rate': ['mean', 'std'],
        'miss_rate': ['mean', 'std'],
        'elephant_rate': ['mean', 'std'],
        'hit_rate_last_10': ['mean', 'std']
    }).round(4)
    print(summary)
    
    # Проверка гипотезы
    hit_means = final_df.groupby('mode')['hit_rate'].mean()
    hit_std = hit_means.std()
    
    print("\n" + "=" * 70)
    print("📜 ПРОВЕРКА ЗАКОНА СОКОЛА")
    print("=" * 70)
    
    print(f"\nСредний hit_rate по режимам:")
    for mode, hr in hit_means.items():
        print(f"  {mode}: {hr:.4f}")
    
    print(f"\nРазброс между режимами (std): {hit_std:.4f}")
    
    if hit_std < 0.05:
        print("\n✅ ЗАКОН СОКОЛА ПОДТВЕРЖДЁН!")
        print(f"   Аттрактор ELIZA: hit_rate = {hit_means.mean():.3f} ± {hit_std:.3f}")
        print("   → Доля осмысленных ответов не зависит от собеседника")
    else:
        print("\n⚠️ ЗАКОН СОКОЛА НЕ ПОДТВЕРЖДАЕТСЯ")
        print("   → hit_rate зависит от типа собеседника")
    
    # Анализ "купи слона"
    elephant_means = final_df.groupby('mode')['elephant_rate'].mean()
    print(f"\n🐘 'Купи слона' (бесконечные уточнения):")
    for mode, er in elephant_means.items():
        print(f"  {mode}: {er:.3f}")
    
    print(f"\n✅ Результаты сохранены: {csv_path}")
    print(f"📊 Всего диалогов: {len(final_df)}")
    print(f"🎭 Всего реплик ELIZA: {len(final_df) * MAX_TURNS:,}")
    
    print("\n" + "=" * 70)
    print("Математик докладывает: эксперимент готов к запуску.")
    print("=" * 70)


if __name__ == "__main__":
    main()
