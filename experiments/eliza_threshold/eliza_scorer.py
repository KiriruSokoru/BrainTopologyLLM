"""
Классификатор ответов ELIZA: попадание / невпопад / купи слона
"""
import re

class ElizaScorer:
    def __init__(self):
        # Паттерны попаданий: ELIZA переспрашивает с использованием ключевых слов
        self.hit_patterns = [
            r'why do you (feel|think|say)',
            r'tell me about your',
            r'what makes you',
            r'how long have you been',
            r'does .* bother you',
            r'why (yes|no)',
            r'you mentioned earlier',
            r'why do you say',
            r'what do you mean by',
            r'can you tell me more about',
            r'i see\. tell me more',
            r'how does that make you feel'
        ]
        
        # Паттерны "купи слона": бесконечное уточнение без прогресса
        self.elephant_patterns = [
            r'^what do you mean by',
            r'^can you explain',
            r'^please clarify',
            r'^define',
            r'in other words',
            r'what does that mean',
            r'i don\'t understand',
            r'could you rephrase'
        ]
        
        # Паттерны невпопад: общие фразы, игнорирующие контекст
        self.miss_patterns = [
            r'^tell me more$',
            r'^i see$',
            r'^please continue$',
            r'^go on$',
            r'^i understand$',
            r'^interesting$',
            r'^why do you say that\?$',
            r'^tell me more\.$'
        ]
    
    def classify(self, eliza_response, user_input=None):
        """
        Возвращает:
        - 'hit': осмысленный ответ
        - 'miss': невпопад
        - 'elephant': бесконечное уточнение ("купи слона")
        """
        response_lower = eliza_response.lower().strip()
        
        # Проверка на "купи слона"
        for pattern in self.elephant_patterns:
            if re.search(pattern, response_lower):
                return 'elephant'
        
        # Проверка на попадание
        for pattern in self.hit_patterns:
            if re.search(pattern, response_lower):
                return 'hit'
        
        # Проверка на невпопад
        for pattern in self.miss_patterns:
            if re.fullmatch(pattern, response_lower):
                return 'miss'
        
        # Если есть user_input — проверим, использовала ли ELIZA слова из него
        if user_input:
            user_words = set(user_input.lower().split())
            response_words = set(response_lower.split())
            if len(user_words & response_words) >= 2:
                return 'hit'
        
        # По умолчанию — невпопад
        return 'miss'
    
    def classify_sequence(self, responses, user_inputs=None):
        """Классифицирует последовательность ответов"""
        classifications = []
        for i, resp in enumerate(responses):
            user_inp = user_inputs[i] if user_inputs and i < len(user_inputs) else None
            classifications.append(self.classify(resp, user_inp))
        return classifications
