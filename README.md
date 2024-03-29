# piskvorky-rl-maturitni-prace
Tento projekt je maturitní práce pro Gymnázium nad Alejí ve školním roce 2022/2023. Jejím účelem bylo vytvořit protivníka v piškvorkách na základě umělé inteligence. Projekt má i grafické rozhraní.

![plot](https://github.com/tad34s/piskvorky-rl-maturitni-prace/blob/main/image.png?raw=true)

# Developer Docs
## How to install
1. git clone the repository
2. Use pip install -r requirements.txt to install the required packages.
3. To run the graphical interface, run main.py in the package interface.
## Architecture
The project contains three packages: piskvorky, interface, and bot. 
- Package `piskvorky` is the game engine. It contains useful functions and the Pisvkorky class.
- `interface` uses PyGame to create a graphical interface for the game.
- The `bot` package contains the Players package. All the player classes use the abstract class Player.
  - All players have the following attributes: name, to_train, and methods: move, new_game, game_end

Module varaibles.py has important global variables. GAME_SIZE can be changed, but networks trained on a different size will not work.
### DQN
This package contains the CNNPlayers, training package, and networks.
We have two types of CNNPlayer: CNNPlayer and CNNPlayer_proximal. The difference is in the implementation of Q-Learing:

- CNNPlayer: It is trained after every match it plays. Targets are computed by the Bellman equation. Memory stays after training.

- CNNPlayer_proximal: Is trained after N matches. Targets are computed by averaging the reward the move led to. Memory is wiped after training.

There are also two networks implemented here:
- CNNetwork_preset: is smaller and has preset kernel weights (5x5), mainly uses convolution, and has better results.

- CNNetwork_big: Bigger has a lot of linear layers.

### AlphaZero
Implements the algorithm AlphaZero. Use the AlphaPlayer to play games.

---
See further documentation in the comments.
