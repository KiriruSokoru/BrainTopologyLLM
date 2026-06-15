"""
ELIZA с человечными багами: забывчивость, сарказм, зависания
"""
from classic_eliza import ClassicELIZA
import random

class NoisyELIZA(ClassicELIZA):
    def __init__(self, noise_level=0.3):
        super().__init__()
        self.noise_level = noise_level
        self.loop_counter = 0
    
    def respond(self, user_input):
        self.loop_counter += 1
        
        # Баг: зависание
        if random.random() < 0.05 * (1 + self.noise_level):
            return "... (silence) ... Hello? Are you there?"
        
        # Баг: сарказм
        if random.random() < 0.1 * self.noise_level and user_input.lower().strip() == "yes":
            return "Oh, 'yes'. Very insightful."
        
        # Баг: забывчивость
        if random.random() < 0.07 * self.noise_level:
            return "I'm sorry, I lost my train of thought. Let's start over. How are you?"
        
        # Баг: циклическое повторение
        if self.loop_counter % 10 == 0 and random.random() < 0.15 * self.noise_level:
            return "I feel like we're going in circles."
        
        return super().respond(user_input)
    
    def reset(self):
        self.loop_counter = 0
