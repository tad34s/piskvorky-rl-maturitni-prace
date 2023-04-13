from piskvorky import Piskvorky
from model import AlphaCNNetwork_preset
from trainer import Trainer
from variables import VELIKOST

game = Piskvorky(VELIKOST)

model = AlphaCNNetwork_preset(game.size,"123",load=True)
if __name__ == '__main__':
    trainer = Trainer(game,"123", model,restrict_movement=True,num_episodes=50)
    trainer.learn()