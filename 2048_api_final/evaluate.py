from game2048.game import Game
from game2048.displays import Display
from game2048.agents import Agent
from keras.models import load_model
import keras
import tensorflow as tf
import numpy as np

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


#NUM_X_CLASSES = 14
class TestAgent(Agent):
    
    def __init__(self, game, display=None):
        self.game = game
        self.display = display
        self.model1 = load_model('my_best_DL_2048_model1.h5')
        self.model2 = load_model('my_best_DL_2048_model2.h5')
        self.model3 = load_model('my_best_DL_2048_model3.h5')
        self.model4 = load_model('my_best_DL_2048_model4.h5')
        self.model5 = load_model('my_best_DL_2048_model5.h5')
        self.model6 = load_model('my_best_DL_2048_model6.h5')
        self.model7 = load_model('my_best_DL_2048_model7.h5')
    
    def step(self):
        
        board = np.log2(np.maximum(np.array(self.game.board), 1)).reshape(1, 4, 4, 1)
        board = keras.utils.np_utils.to_categorical(board, 14)
        direction = []
        direction.extend(np.argmax(self.model1.predict(board), axis=1))
        direction.extend(np.argmax(self.model2.predict(board), axis=1))
        direction.extend(np.argmax(self.model3.predict(board), axis=1))
        direction.extend(np.argmax(self.model4.predict(board), axis=1))
        direction.extend(np.argmax(self.model5.predict(board), axis=1))
        direction.extend(np.argmax(self.model6.predict(board), axis=1))
        direction.extend(np.argmax(self.model7.predict(board), axis=1))
        direction = np.argmax(np.bincount(direction))
        return direction

    
    
if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 10

    '''====================
    Use your own agent here.'''
    '''===================='''

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass=TestAgent)
        scores.append(score)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
