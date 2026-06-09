import torch
from tqdm import tqdm

class AttentionSurgery:
    def __init__(self, threshold_ricci=0.7, noise_scale=0.05):
        self.threshold = threshold_ricci
        self.noise_scale = noise_scale
    
    def apply_to_model(self, model, ricci_scores):
        """
        Хирургия для GPT-2 attention heads
        """
        modified_heads = 0
        
        with tqdm(total=len(model.transformer.h), desc="Хирургия attention") as pbar:
            for layer_idx, block in enumerate(model.transformer.h):
                attn = block.attn
                n_head = attn.num_heads
                head_dim = attn.head_dim
                n_embd = attn.embed_dim
                
                c_attn_weight = attn.c_attn.weight
                q_weight = c_attn_weight[:n_embd, :].clone()
                k_weight = c_attn_weight[n_embd:2*n_embd, :].clone()
                v_weight = c_attn_weight[2*n_embd:, :].clone()
                
                # Находим здоровые и сингулярные головы
                healthy_mask = ricci_scores[layer_idx] < self.threshold
                singular_mask = ricci_scores[layer_idx] >= self.threshold
                
                healthy_indices = torch.where(healthy_mask)[0]
                singular_indices = torch.where(singular_mask)[0]
                
                if len(singular_indices) == 0:
                    pbar.update(1)
                    continue
                
                # Если здоровых голов нет - используем все головы как базу
                if len(healthy_indices) < 2:
                    # Берём все головы, но исключаем самые сингулярные
                    all_indices = torch.arange(n_head)
                    sorted_scores, sorted_idx = torch.sort(ricci_scores[layer_idx])
                    # Берём 50% наименее сингулярных голов
                    healthy_indices = sorted_idx[:max(2, n_head//2)]
                    print(f"⚠️ Слой {layer_idx}: мало здоровых голов, используем {len(healthy_indices)} наименее сингулярных")
                
                # Собираем "колпачок" - среднее здоровых голов
                healthy_q_segments = []
                healthy_k_segments = []
                healthy_v_segments = []
                
                for h in healthy_indices:
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    healthy_q_segments.append(q_weight[:, start:end])
                    healthy_k_segments.append(k_weight[:, start:end])
                    healthy_v_segments.append(v_weight[:, start:end])
                
                avg_q = torch.stack(healthy_q_segments).mean(dim=0)
                avg_k = torch.stack(healthy_k_segments).mean(dim=0)
                avg_v = torch.stack(healthy_v_segments).mean(dim=0)
                
                # Лечим сингулярные головы
                for h in singular_indices:
                    start = h * head_dim
                    end = (h + 1) * head_dim
                    
                    with torch.no_grad():
                        # Заменяем на колпачок + шум
                        q_weight[:, start:end] = avg_q + self.noise_scale * torch.randn_like(avg_q)
                        k_weight[:, start:end] = avg_k + self.noise_scale * torch.randn_like(avg_k)
                        v_weight[:, start:end] = avg_v + self.noise_scale * torch.randn_like(avg_v)
                    
                    modified_heads += 1
                
                # Собираем обратно в c_attn
                new_c_attn_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                attn.c_attn.weight.data = new_c_attn_weight
                
                pbar.update(1)
                pbar.set_postfix({'modified': modified_heads, 'healthy': len(healthy_indices)})
        
        print(f"✅ Attention хирургия: изменено {modified_heads} голов")
        return model
