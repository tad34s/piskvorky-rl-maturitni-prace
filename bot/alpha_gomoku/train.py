from piskvorky import Piskvorky
from model import AlphaCNNetwork_preset
from trainer import Trainer
from variables import VELIKOST

game = Piskvorky(VELIKOST)


trainer = Trainer(game,"123",restrict_movement=True)
trainer.learn()