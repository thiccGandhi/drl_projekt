# drl_projekt
Deep Reinforcement Learning Projekt with Jesus

![Jesus](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExNGd5dGJicTgxOWNlZ3N1cWhmOGlzeDJueDMyMGlmOGp5MHRqcjk2byZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6sjfJk4cz9ei4/giphy.gif)

Algorithms to implement: DDPG (optionally TD3, SAC)

https://spinningup.openai.com/en/latest/algorithms/ddpg.html

oder hier noch das original DDPG paper

https://arxiv.org/abs/1509.02971

TD3 paper

https://arxiv.org/abs/1802.09477


---

`main.py` wird alles initialisiert was benötigt wird (Actor, Critic, ReplayBuffer, etc.)\
Es wird alles in einem `Agent`-Objekt zusammengeführt (DDPG oder TD3 bis jetzt, files in `agents\`)\
Dann wird der `Trainer` (`trainer\trainer.py`) aufgerufen, welcher in `train()` über die Episoden iteriert.\
In jeder Episode wird dann `agent.update()` aufgerufen (Update von Actor, Critic und Targets)\
