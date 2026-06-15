"""
ELIZA Classic (Weizenbaum, 1966) - терапевтический режим
Паттерны: повтор фразы, переформулирование, открытые вопросы
"""
import re
import random

class ClassicELIZA:
    def __init__(self):
        self.patterns = {
            r"my (.*?) (?:is|are) (.*)": [
                "Why do you say your {} is {}?",
                "Tell me more about your {} being {}."
            ],
            r"i feel (.*)": [
                "Why do you feel {}?",
                "Does feeling {} bother you often?"
            ],
            r"i think (.*)": [
                "What makes you think {}?",
                "Do you often think about {}?"
            ],
            r"you (.*) me": [
                "Why do you say I {} you?",
                "What makes you think I {} you?"
            ],
            r"i am (.*)": [
                "Why are you {}?",
                "How long have you been {}?"
            ],
            r"because (.*)": [
                "Is that the real reason?",
                "What other reasons might there be?"
            ],
            r"yes": ["I see. Tell me more.", "Why yes?", "Go on."],
            r"no": ["Why not?", "Why do you say no?", "Tell me more."],
            r"mother": ["Tell me about your mother.", "How do you feel about your mother?"],
            r"father": ["Tell me about your father.", "How does your father influence you?"],
            r"(.*)": ["Tell me more.", "I see.", "Please continue.", "Why do you say that?"]
        }
    
    def respond(self, user_input):
        user_input = user_input.lower().strip()
        for pattern, responses in self.patterns.items():
            match = re.search(pattern, user_input)
            if match:
                if pattern.startswith(r"my"):
                    response = random.choice(responses).format(match.group(1), match.group(2))
                elif pattern.startswith(r"i feel") or pattern.startswith(r"i think") or pattern.startswith(r"i am"):
                    response = random.choice(responses).format(match.group(1))
                elif pattern.startswith(r"you (.*) me"):
                    response = random.choice(responses).format(match.group(1))
                else:
                    response = random.choice(responses)
                return response
        return random.choice(self.patterns[r"(.*)"])
    
    def reset(self):
        pass
