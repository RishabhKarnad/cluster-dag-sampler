import jax.random as random


class RandomState:
    def __init__(self, seed=678):
        self.key = random.PRNGKey(seed)

    def get_key(self):
        self.key, subk = random.split(self.key)
        return subk

    def get_random_number(self):
        self.key, subk = random.split(self.key)
        return subk[0].item()

    def set_key(self, *, seed):
        self.key = random.PRNGKey(seed)


random_state = RandomState()
