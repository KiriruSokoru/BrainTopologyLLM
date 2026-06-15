"""
Анализатор аттрактора для ELIZA
Метрика: hit_rate (доля осмысленных ответов) — аттрактор, если стабилизируется
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from eliza_scorer import ElizaScorer


class AttractorAnalyzer:
    def __init__(self, eliza_instance, user_model, max_turns=100):
        self.eliza = eliza_instance
        self.user_model = user_model
        self.max_turns = max_turns
        self.scorer = ElizaScorer()
    
    def run_dialogue(self, start_phrase):
        """Запускает диалог и возвращает семантическую статистику"""
        self.eliza.reset()
        if hasattr(self.user_model, 'reset'):
            self.user_model.reset()
        
        user_input = start_phrase
        responses = []
        user_inputs = [start_phrase]
        classifications = []
        hit_count = 0
        miss_count = 0
        elephant_count = 0
        
        for turn in range(self.max_turns):
            response = self.eliza.respond(user_input)
            responses.append(response)
            
            # Классифицируем ответ
            cls = self.scorer.classify(response, user_input)
            classifications.append(cls)
            
            if cls == 'hit':
                hit_count += 1
            elif cls == 'miss':
                miss_count += 1
            else:  # elephant
                elephant_count += 1
            
            # Если 3 "купи слона" подряд — бесконечное уточнение
            if len(classifications) >= 3 and classifications[-1] == classifications[-2] == classifications[-3] == 'elephant':
                break
            
            # Генерируем следующий ответ пользователя
            user_input = self.user_model.respond(response, responses)
            user_inputs.append(user_input)
        
        total = len(classifications)
        return {
            'responses': responses,
            'classifications': classifications,
            'hit_rate': hit_count / total if total > 0 else 0,
            'miss_rate': miss_count / total if total > 0 else 0,
            'elephant_rate': elephant_count / total if total > 0 else 0,
            'total_turns': total,
            'final_classification': classifications[-1] if classifications else 'miss',
            'hit_rate_last_10': self._compute_last_n_rate(classifications, 'hit', 10),
            'miss_rate_last_10': self._compute_last_n_rate(classifications, 'miss', 10),
            'elephant_rate_last_10': self._compute_last_n_rate(classifications, 'elephant', 10)
        }
    
    def _compute_last_n_rate(self, classifications, cls_type, n=10):
        """Доля определённого типа в последних n ответах"""
        last_n = classifications[-n:]
        return last_n.count(cls_type) / len(last_n) if last_n else 0
    
    def run_experiment(self, start_phrases, repeats_per_phrase=48, verbose=True):
        """Запускает полный эксперимент"""
        results = []
        total = len(start_phrases) * repeats_per_phrase
        
        if verbose:
            iterator = tqdm(start_phrases, desc="Фразы", unit="phrase")
        else:
            iterator = start_phrases
        
        for phrase_idx, phrase in enumerate(iterator):
            for rep in range(repeats_per_phrase):
                outcome = self.run_dialogue(phrase)
                
                result = {
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
        
        return pd.DataFrame(results)
