import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2Config, GPT2LMHeadModel
from data.copy_task import create_dataloaders
from surgery.attention_surgery import AttentionSurgery
from train.trainer import train_model
import yaml
import numpy as np

def inject_singularity(model, damage_level):
    """
    Создаём сингулярность: заставляем голову давать детерминированный вывод
    """
    ricci_scores = torch.zeros(len(model.transformer.h), model.config.n_head)
    
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.transformer.h):
            attn = layer.attn
            n_head = attn.num_heads
            head_dim = attn.head_dim
            n_embd = attn.embed_dim
            
            c_attn_weight = attn.c_attn.weight
            q_weight = c_attn_weight[:n_embd, :].clone()
            k_weight = c_attn_weight[n_embd:2*n_embd, :].clone()
            
            damaged_heads = []
            for h in range(n_head):
                if np.random.random() < damage_level:
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    
                    # Делаем Q и K почти одинаковыми (низкая энтропия = залипание)
                    q_weight[:, start:end] = torch.ones_like(q_weight[:, start:end]) * 0.01
                    k_weight[:, start:end] = torch.ones_like(k_weight[:, start:end]) * 0.01
                    
                    damaged_heads.append(h)
                    ricci_scores[layer_idx, h] = 0.9  # Высокая сингулярность
            
            # Собираем обратно
            new_c_attn_weight = torch.cat([q_weight, k_weight, c_attn_weight[2*n_embd:, :]], dim=0)
            attn.c_attn.weight.data = new_c_attn_weight
            
            if damaged_heads:
                print(f"  Layer {layer_idx}: сингулярные головы {damaged_heads} (score=0.9)")
                # Здоровым головам - низкая сингулярность
                for h in range(n_head):
                    if h not in damaged_heads:
                        ricci_scores[layer_idx, h] = 0.1
    
    return ricci_scores

def main():
    with open('config/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    train_loader, val_loader = create_dataloaders(config)
    print(f"📦 Данные: {len(train_loader)} батчей тренировка, {len(val_loader)} валидация")
    
    # Тестируем только один damage level для быстрой отладки
    damage_levels = [0.3, 0.5, 0.7]
    results = {'with_surgery': {}, 'without_surgery': {}}
    
    for surgery_enabled in [True, False]:
        print(f"\n{'='*60}")
        print(f"🔬 ЭКСПЕРИМЕНТ: Хирургия = {surgery_enabled}")
        print(f"{'='*60}")
        
        for damage in damage_levels:
            print(f"\n  📊 Damage level = {damage}")
            
            # Создаём модель
            gpt_config = GPT2Config(
                n_layer=config['model']['n_layer'],
                n_head=config['model']['n_head'],
                n_embd=config['model']['n_embd'],
                max_position_embeddings=config['model']['max_position_embeddings']
            )
            model = GPT2LMHeadModel(gpt_config).to(device)
            
            # Внедряем сингулярность
            ricci_scores = inject_singularity(model, damage)
            
            if surgery_enabled:
                print(f"    🔪 Применяем хирургию Перельмана")
                surgery = AttentionSurgery(
                    threshold_ricci=config['surgery']['threshold_ricci'],
                    noise_scale=config['surgery']['noise_scale']
                )
                model = surgery.apply_to_model(model, ricci_scores)
            else:
                print(f"    ⚠️ Контрольная группа (без хирургии)")
            
            # Обучение
            print(f"    🚀 Обучение {config['training']['epochs']} эпох...")
            final_loss = train_model(
                model, train_loader, val_loader, config, device,
                damage_info=f"surgery_{surgery_enabled}_damage_{damage}"
            )
            
            if surgery_enabled:
                results['with_surgery'][damage] = final_loss
            else:
                results['without_surgery'][damage] = final_loss
            
            print(f"    ✅ Финальная loss: {final_loss:.4f}")
    
    # Анализ
    print("\n" + "="*60)
    print("📊 АНАЛИЗ АТТРАКТОРА")
    print("="*60)
    
    print("\nБез хирургии (контроль):")
    for d, l in results['without_surgery'].items():
        print(f"  damage={d}: loss={l:.4f}")
    
    print("\nС хирургией Перельмана:")
    for d, l in results['with_surgery'].items():
        print(f"  damage={d}: loss={l:.4f}")
    
    if results['without_surgery'] and results['with_surgery']:
        var_without = np.var(list(results['without_surgery'].values()))
        var_with = np.var(list(results['with_surgery'].values()))
        
        print(f"\n📈 Дисперсия финальных потерь:")
        print(f"  Без хирургии: {var_without:.6f}")
        print(f"  С хирургией: {var_with:.6f}")
        
        if var_with < var_without and var_with < 0.01:
            print("\n🎉"*20)
            print("✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА!")
            print("🎉"*20)
        elif var_with < var_without:
            print("\n📈"*20)
            print("⚠️ ТЕНДЕНЦИЯ ЕСТЬ, НО НУЖНО БОЛЬШЕ ЭПОХ")
            print("📈"*20)
        else:
            print("\n❌"*20)
            print("ГИПОТЕЗА НЕ ПОДТВЕРЖДЕНА")
            print("❌"*20)

if __name__ == "__main__":
    main()
