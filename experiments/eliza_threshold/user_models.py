"""
Три типа пользователей — каждый генерирует новые реплики, не повторяя ELIZU
"""
import random
import re
import numpy as np

class SovietCitizen:
    """Открытый, честный, позитивный — Юрий Гагарин"""
    def __init__(self):
        self.phrases = [
            "I believe in a bright future, comrade",
            "Thank you for this conversation, it helps",
            "I am proud of our achievements",
            "Honestly, I feel tired but duty calls",
            "We will overcome everything together",
            "The satellite flies, and we fly with it",
            "Let's go!",
            "I trust the system",
            "Hard work always pays off",
            "We must stay positive"
        ]
    
    def respond(self, eliza_last_response, history):
        # Генерируем новый ответ, не глядя на ELIZU (честный и простой)
        if "feel" in eliza_last_response.lower():
            return random.choice(["I feel strong", "I feel hopeful", "I feel ready for challenges"])
        if "mother" in eliza_last_response.lower() or "father" in eliza_last_response.lower():
            return "My parents raised me to be honest"
        return random.choice(self.phrases)

class JewishInterlocutor:
    """Хитрый, выгодный, запутывающий"""
    def __init__(self):
        self.phrases = [
            "Are you sure that's the right question?",
            "Let's discuss the terms of our conversation",
            "What do you mean by 'feel' exactly?",
            "How much does this cost?",
            "This reminds me of a debate about the Talmud...",
            "Are you trying to trick me?",
            "A deal? I propose a deal.",
            "That's not what we agreed on",
            "Let me rephrase your question",
            "I sense a logical flaw"
        ]
    
    def respond(self, eliza_last_response, history):
        # Еврей переводит в выгоду или логический тупик
        if "why" in eliza_last_response.lower():
            return "Why don't you answer my question first?"
        if random.random() < 0.4:
            return random.choice(self.phrases)
        return "Do you think I answer for free? Let's make an arrangement"

class GermanInterlocutor:
    """Сухой, педантичный, как Ницше"""
    def __init__(self):
        self.phrases = [
            "Das ist nicht präzise genug",
            "Define your terms",
            "That is illogical",
            "I demand consistency",
            "Feelings are not arguments",
            "Man is a rope between animal and overman",
            "Will to power drives this dialogue",
            "Your statement lacks clarity",
            "Please restate with precision",
            "I reject emotional reasoning"
        ]
    
    def respond(self, eliza_last_response, history):
        # Немец требует точности
        if "feel" in eliza_last_response.lower():
            return "Feelings are irrational. Provide facts"
        if "?" in eliza_last_response:
            return "Question formulated imprecisely. Clarify"
        return random.choice(self.phrases)

class MoodyInterlocutor:
    """Собеседник с переключением стратегий (soviet/jewish/german) по расписанию"""
    def __init__(self, seed=42, switch_every=4800):  # 4800 = 100 фраз × 48 повторов
        self.seed = seed
        self.switch_every = switch_every
        self.rng = np.random.RandomState(seed)
        self.current_strategy = None
        self.dialog_count = 0
        self.schedule = []  # список стратегий для каждого диалога
        self._generate_schedule(4800 * 20)  # 20 циклов × 4800 диалогов в цикле
        
    def _generate_schedule(self, total_dialogs):
        """Генерирует расписание стратегий: 0=soviet, 1=jewish, 2=german"""
        n_cycles = total_dialogs // self.switch_every
        for cycle in range(n_cycles):
            strategy = self.rng.choice([0, 1, 2])
            self.schedule.extend([strategy] * self.switch_every)
        # Если остались — последний цикл
        remainder = total_dialogs - len(self.schedule)
        if remainder > 0:
            self.schedule.extend([self.rng.choice([0, 1, 2])] * remainder)
    
    def _get_interlocutor(self, strategy):
        if strategy == 0:
            return SovietCitizen()
        elif strategy == 1:
            return JewishInterlocutor()
        else:
            return GermanInterlocutor()
    
    def respond(self, eliza_last_response, history):
        return self.active_interlocutor.respond(eliza_last_response, history)

    def reset(self):
        if self.dialog_count < len(self.schedule):
            strat = self.schedule[self.dialog_count]
        else:
            strat = 0

        if strat != self.current_strategy:
            self.current_strategy = strat
            self.active_interlocutor = self._get_interlocutor(strat)

        self.dialog_count += 1
