from game2048.agents import Agent
import keras
from keras.models import load_model
import numpy as np

class MyAgent1(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.model = load_model('my_model.h5')

    def step(self):
        board = np.log2(np.maximum(np.array(self.game.board), 1)).reshape(1, 4, 4, 1)
        board = keras.utils.np_utils.to_categorical(board, 12)
        direction = self.model.predict_classes(board)[0]
        return direction
