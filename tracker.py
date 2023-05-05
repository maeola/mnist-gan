class Tracker:
    def __init__(self, gamma):
        self.gamma = gamma
        self.value = 0
        self.count = 0

    def __repr__(self):
        return f'{self.value:6.3f}'

    def add(self, x):
        self.count += 1
        gamma = max(1 / self.count, self.gamma)
        self.value = self.value * (1 - gamma) + x * gamma

