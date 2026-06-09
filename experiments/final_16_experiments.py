import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Config, GPT2LMHeadModel
from data.copy_task import create_dataloaders
from surgery.attention_surgery import AttentionSurgery
from train.trainer import train_model
from train.trickster import Trickster
import yaml
import numpy as np
import json
from datetime import datetime

def inject_autism(model, damage_level):
    """Создаём аутистов (залипшие головы)"""
    autistic_heads = []
    
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.transformer.h):
            attn = layer.attn
            n_head = attn.num_heads
            head_dim = attn.head_dim
            n_embd = attn.embed_dim
            
            c_attn_weight = attn.c_attn.weight
            q_weight = c_attn_weight[:n_embd, :].clone()
            k_weight = c_attn_weight[n_embd:2*n_embd, :].clone()
            
            for h in range(n_head):
                if np.random.random() < damage_level:
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    
                    # Аутист: все значения Q и K одинаковые (залипание)
                    q_weight[:, start:end] = torch.ones_like(q_weight[:, start:end]) * 0.01
                    k_weight[:, start:end] = torch.ones_like(k_weight[:, start:end]) * 0.01
                    autistic_heads.append((layer_idx, h))
            
            new_c_attn_weight = torch.cat([q_weight, k_weight, c_attn_weight[2*n_embd:, :]], dim=0)
            attn.c_attn.weight.data = new_c_attn_weight
    
    print(f"    🧩 Аутистов создано: {len(autistic_heads)}")
    return autistic_heads

def run_experiment(config, damage, trickster_strength, run_id):
    """Запуск одного эксперимента"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создаём модель
    gpt_config = GPT2Config(
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embd=config['model']['n_embd'],
        max_position_embeddings=config['model']['max_position_embeddings']
    )
    model = GPT2LMHeadModel(gpt_config).to(device)
    
    # Заражаем аутизмом
    autistic_heads = inject_autism(model, damage)
    
    # Создаём фальшивые ricci_scores для хирургии
    ricci_scores = torch.ones(len(model.transformer.h), model.config.n_head) * 0.1
    for layer, head in autistic_heads:
        ricci_scores[layer, head] = 0.9
    
    # Применяем хирургию
    surgery = AttentionSurgery(
        threshold_ricci=config['surgery']['threshold_ricci'],
        noise_scale=config['surgery']['noise_scale']
    )
    model = surgery.apply_to_model(model, ricci_scores)
    
    # Создаём даталоудеры
    train_loader, val_loader = create_dataloaders(config)
    
    # Добавляем Плута в тренер (через глобальную переменную)
    from train.trickster import active_trickster
    active_trickster.strength = trickster_strength
    active_trickster.enabled = trickster_strength > 0
    
    # Обучаем
    final_loss = train_model(
        model, train_loader, val_loader, config, device,
        damage_info=f"damage_{damage}_trickster_{trickster_strength}_run_{run_id}"
    )
    
    return final_loss

def main():
    with open('config/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Параметры эксперимента
    damage_levels = [0.1, 0.3, 0.5, 0.7]
    trickster_levels = [0.0, 0.05, 0.1, 0.2]
    n_runs = 3  # повторов для статистики
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'damage_levels': damage_levels,
            'trickster_levels': trickster_levels,
            'n_runs': n_runs
        },
        'experiments': []
    }
    
    total_exp = len(damage_levels) * len(trickster_levels) * n_runs
    current_exp = 0
    
    print("="*60)
    print("🧪 ЗАПУСК 16 ЭКСПЕРИМЕНТОВ (с повторами)")
    print(f"📊 Всего запусков: {total_exp}")
    print("="*60)
    
    for damage in damage_levels:
        for trickster in trickster_levels:
            exp_losses = []
            
            for run in range(n_runs):
                current_exp += 1
                print(f"\n[{current_exp}/{total_exp}] damage={damage}, trickster={trickster}, run={run+1}")
                
                loss = run_experiment(config, damage, trickster, run+1)
                exp_losses.append(loss)
                
                print(f"    → Финальная loss: {loss:.4f}")
            
            # Сохраняем статистику по этой комбинации
            exp_result = {
                'damage': damage,
                'trickster_strength': trickster,
                'losses': exp_losses,
                'mean_loss': np.mean(exp_losses),
                'std_loss': np.std(exp_losses),
                'var_loss': np.var(exp_losses)
            }
            results['experiments'].append(exp_result)
            
            print(f"  📊 Статистика: mean={exp_result['mean_loss']:.4f}, std={exp_result['std_loss']:.6f}")
    
    # Сохраняем результаты
    with open('results_final_16.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Финальный анализ
    print("\n" + "="*60)
    print("📊 ФИНАЛЬНЫЙ АНАЛИЗ АТТРАКТОРА")
    print("="*60)
    
    # Группируем по trickster strength
    for trickster in trickster_levels:
        relevant = [e for e in results['experiments'] if e['trickster_strength'] == trickster]
        losses = [e['mean_loss'] for e in relevant]
        
        print(f"\n🎭 Плут силой {trickster}:")
        print(f"   Потери при разных damage: {[f'{l:.4f}' for l in losses]}")
        print(f"   Дисперсия: {np.var(losses):.6f}")
        
        if np.var(losses) < 0.01:
            print(f"   ✅ Аттрактор достигнут")
        elif np.var(losses) < 0.05:
            print(f"   ⚠️ Частичная сходимость")
        else:
            print(f"   ❌ Система развалилась")
    
    print("\n" + "="*60)
    print("✅ Эксперименты завершены")
    print(f"📁 Результаты сохранены в results_final_16.json")
    print("="*60)
    
    return results

if __name__ == "__main__":
    main()
