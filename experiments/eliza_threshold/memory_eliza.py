"""
ELIZA с памятью на 5 последних реплик
"""
from classic_eliza import ClassicELIZA
import random

class MemoryELIZA(ClassicELIZA):
    def __init__(self, memory_size=5):
        super().__init__()
        self.memory = []
        self.memory_size = memory_size
    
    def respond(self, user_input):
        self.memory.append(user_input)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        # Зацикливание?
        if len(self.memory) >= 3 and len(set(self.memory[-3:])) == 1:
            return "We've been over this. What else is on your mind?"
        
        response = super().respond(user_input)
        
        # Ссылка на память
        if len(self.memory) >= 2 and random.random() < 0.1:
            earlier = self.memory[-2]
            return f"You mentioned earlier '{earlier}'. {response.lower()}"
        
        return response
    
    def reset(self):
        self.memory = []
