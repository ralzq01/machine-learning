## FindPath

FindPaht is a game aiming to find the treasure on the map. However, while walking on the map,agent may get into a trap and thus failed the game.

Every time the agent will start at a random position, also the map is invisible for agent. So agent must try a lot to find a optimal way to find the treasure. Every move will cost agent 1 bonus and after find the treasure, agent will get a certain bonus. The goal of this game is to get as higher bonus as posible.

### Methods

In this program, we use Reinforcement Learning to train the agent to figure out a best way to find the treasure (a.k.a get the highest bonus on the game). This game is easy, only using Q-learning we can achieve the goal.


### Prerequest for running

For runing this code, you should at least have these environments:

* Python 2.7.*

* gym (for installing: pip install gym)

* numpy (for installing: pip install numpy)

### How to run

Under directory of `findpath/` , simply run the command `$ python run.py`

You will see the training result in a GUI where 10 games will start one by one, and you can see the movement of the agent to acheive the highest bonus.

