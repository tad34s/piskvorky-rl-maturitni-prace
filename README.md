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
- The bot package is further devided into individual players and 2 packages: AlphaZero and DQN. All the player classes use the abstract class Player.

