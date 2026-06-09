import torch
import random
import numpy as np

class Trickster:
    """Плут — случайный манипулятор для проверки робастности"""
    
    def __init__(self, strength=0.1, frequency=0.3):
        self.strength = strength
        self.frequency = frequency
        self.enabled = strength > 0
        self.tricks_done = 0
        self.last_trick = None
        
    def play_trick(self, model, optimizer=None):
        """Сыграть случайную пакость"""
        if not self.enabled:
            return None
        
        if random.random() > self.frequency:
            return None
        
        tricks = [
            self._swap_weights,
            self._invert_weight,
            self._add_noise,
            self._shuffle_gradients
        ]
        
        trick = random.choice(tricks)
        result = trick(model, optimizer)
        self.tricks_done += 1
        self.last_trick = result
        
        return result
    
    def _swap_weights(self, model, optimizer=None):
        """Меняем местами два случайных нейрона"""
        with torch.no_grad():
            layer = random.choice(model.transformer.h)
            # Меняем в FFN
            ffn = layer.mlp
            w1 = ffn.c_fc.weight
            if w1.size(0) > 1:
                idx1, idx2 = random.sample(range(w1.size(0)), 2)
                w1[idx1], w1[idx2] = w1[idx2].clone(), w1[idx1].clone()
        return "swap_weights"
    
    def _invert_weight(self, model, optimizer=None):
        """Инвертируем знак случайного веса"""
        with torch.no_grad():
            layer = random.choice(model.transformer.h)
            param = random.choice([layer.attn.c_attn.weight, layer.mlp.c_fc.weight])
            idx = random.randint(0, param.numel() - 1)
            param.view(-1)[idx] *= -1
        return "invert_weight"
    
    def _add_noise(self, model, optimizer=None):
        """Добавляем шум в случайный слой"""
        with torch.no_grad():
            layer = random.choice(model.transformer.h)
            param = random.choice([layer.attn.c_attn.weight, layer.mlp.c_fc.weight])
            noise = torch.randn_like(param) * self.strength
            param.add_(noise)
        return "add_noise"
    
    def _shuffle_gradients(self, model, optimizer):
        """Перемешиваем градиенты перед обновлением"""
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None and random.random() < 0.1:
                        # Перемешиваем градиенты
                        grad_flat = param.grad.view(-1)
                        perm = torch.randperm(grad_flat.size(0))
                        grad_flat[:] = grad_flat[perm]
        return "shuffle_gradients"

# Глобальный экземпляр для использования в trainer
active_trickster = Trickster()
