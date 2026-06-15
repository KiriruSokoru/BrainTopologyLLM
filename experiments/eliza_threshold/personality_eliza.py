"""
ELIZA с переключением между 3 личностями:
0 - Терапевт, 1 - Саркастичный, 2 - Восторженный
"""
from classic_eliza import ClassicELIZA
import random
import re

class PersonalityELIZA(ClassicELIZA):
    def __init__(self, switch_probability=0.1):
        super().__init__()
        self.personality = 0
        self.switch_probability = switch_probability
        
        self.sarcastic_responses = {
            r"i feel (.*)": ["Oh, feelings. How original.", "And how does that feeling serve you?"],
            r"i think (.*)": ["Thinking? That's a first.", "Interesting. Wrong, but interesting."],
            r"yes": ["Congratulations, you can agree.", "Next brilliant insight?"],
            r"no": ["Rebel. I like it.", "How contrarian of you."],
            r"(.*)": ["Fascinating. Not really, but go on.", "Sure. Whatever you say."]
        }
        
        self.optimist_responses = {
            r"i feel (.*)": ["That's wonderful! Tell me more!", "Feelings are amazing!"],
            r"i think (.*)": ["Great thinking! {} is fantastic!", "I love how your mind works!"],
            r"yes": ["Yes! Absolutely yes!", "Wonderful!"],
            r"no": ["No is just the beginning of yes!", "That's okay!"],
            r"(.*)": ["Amazing! Please continue!", "I'm so excited to hear more!"]
        }
    
    def respond(self, user_input):
        if random.random() < self.switch_probability:
            self.personality = random.choice([0, 1, 2])
        
        user_input = user_input.lower().strip()
        
        if self.personality == 1:  # Сарказм
            for pattern, responses in self.sarcastic_responses.items():
                if re.search(pattern, user_input):
                    return random.choice(responses)
        elif self.personality == 2:  # Оптимист
            for pattern, responses in self.optimist_responses.items():
                match = re.search(pattern, user_input)
                if match:
                    if pattern.startswith(r"i feel") or pattern.startswith(r"i think"):
                        return random.choice(responses).format(match.group(1))
                    return random.choice(responses)
        
        return super().respond(user_input)
    
    def reset(self):
        self.personality = 0
