# piskvorky-rl-maturitni-prace
Tento projekt je maturitní práce pro Gymnázium nad Alejí ve školním roce 2022/20223. Jejím účelem bylo vytvořit protivníka na základě umělé inteligence.
# Developer Docs
## How to install
1. git clone the repository
2. Use pip install -r requirements.txt to install required packages
3. To run the graphical interface run the main.py in package interface
## Architecture
The projects contains 3 packages: piskvorky, interface and bot. 
- Package piskvorky is the game engine. It contains useful functions and the Pisvkorky class.
- Interface uses PyGame to create an graphical interface for the game.
- The bot package contains the Players. All the player classes use the abstract class Player.
  - All players have attributes: name, to_train and methods: move, new_game, game_end

Module varaibles has important global variables. GAME_SIZE can be changed, but networks trained on different size will not work.
### DQN
This packages contains the CNNPlayers, training package and networks.
We have two typec of CNNPlayer: CNNPlayer and CNNPlayer_proximal. The differrence is in the implementation of Q-Learing:

- CNNPlayer: Is trained after every match it plays. Targets are computed by the Bellman equation. Memory stays after training.

- CNNPlayer_proximal: Is trained after N matches. Targets are computed by averaging the the reward the move led to. Memory is wiped after training.

There are also two networks implemented here:
- CNNetwork_preset: Is smaller and has preset kernel weights (5x5) - mainly uses convolution, has better results

- CNNetwork_big: Bigger has a lot of linear layers

### AlphaZero
Implements the algorithm AlphaZero. Use the AlphaPlayer to play games.

---
See further documentation in the comments
