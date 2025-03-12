import pickle

from promptolution.callbacks import Callback


class PickleCallback(Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.count = 0

    def on_step_end(self, optimizer):
        self.count += 1
        with open(self.output_dir + self.count + ".pickle", "wb") as f:
            pickle.dump(optimizer, f)

        return True
