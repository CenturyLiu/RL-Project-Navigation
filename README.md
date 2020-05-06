# RL-Project-Navigation
Train an agent by value-based reinforcement learning to navigate in a large world and collect yellow bananas, while avoiding blue bananas

## Part 0: Basic reinforcement learning concept involved
![](https://github.com/CenturyLiu/RL-Project-Navigation/blob/master/rl-basic.png)
> The agent-environment interaction in Reinforcement Learning.(Source: [Sutton and Barto, 2017](http://incompleteideas.net/book/RLbook2020.pdf))

In short, the simplest reinforcment learning framework involves an agent interacting with the environment. At each time step, the agent observes the **state** of the environment, choose valid **action** and receive **reward** from the environment. The interaction between the agent and the environment forms a sequence of S<sub>0</sub> ,A<sub>0</sub>, R<sub>1</sub> ,S<sub>1</sub> ,A<sub>1</sub>, R<sub>2</sub>,...,S<sub>T</sub> ,A<sub>T</sub>, R<sub>T+1</sub>       (s - states,a - action,r - reward). The goal of the agent is to learn to choose action to maximize the total reward it can receive. The total reward at time step t is given by G<sub>t</sub> = R<sub>t+1</sub> + R<sub>t+2</sub> + R<sub>t+3</sub> + ... Intuitively, agent should give more emphasis on getting reward in the near future, and pay less attention to the reward in the far future, we introduce a **discount rate γ (gamma)**. The total reward with disount rate is calculated by G<sub>t</sub> = γ\*R<sub>t+1</sub> + γ\*R<sub>t+2</sub> + γ\*R<sub>t+3</sub> + ...

Based on whether the interation has well-defined end, reinforcement learning tasks can be divided in to episodic tasks and continuing tasks. Episodic tasks have well-defined begin and end point (eg. Drive a car from your position to destination). Continuing tasks will run forever (eg. The car is running on an endless road). This readme is discussing my solution to an episodic task.

Mathematically, reinforcement learning can be described as a **Markov Decision Process (MDP)**, which involves **state space S**, **action space A** and **reward R**. For more detailed description of this process, I recommend you take a course for reinforcement learning (eg: [Udacity deep reinforcement learning nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)) or refer to the book [Sutton and Barto, 2017](http://incompleteideas.net/book/RLbook2020.pdf).  

## Part 1: Problem to solve
   - Main goal                                                                                                                                            
     Train an agent to navigate in a large world, collecting yellow bananas while avoiding blue bananas
   - State space                                                                                                           
     The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
   - Action space
     The agent has 4 valid action:
     - `0` move forward
     - `1` move backward
     - `2` turn left
     - `3` turn right
   - Reward setup
     | Banana Collected | Reward |
     |  --------------  | ------ |
     | Yellow           | +1     |
     | Blue             | -1     |
   - End condition for each episode                                                                                       
     The game is timed and the environment will automatically stop the episode once time is over.
   - Requirement for solving the task                                                                                      
     The task is considered solved when the agent gets an average score of +13 over 100 consecutive episodes. 
## Part 2: Idea for solving the problem
The agent I used for solving the problem is a mixture of double DQN, dueling DQN, and priotized experience replay.


references: [deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), [double DQN](https://arxiv.org/abs/1509.06461) , [dueling DQN](https://arxiv.org/abs/1511.06581) , [priotized experience replay](https://arxiv.org/abs/1511.05952).

## Part 3: Project implementation
   - Prerequisite                                                                                                             
     To run the codes in this repository, please follow instruction on [Udacity deep reinforcement learning](https://github.com/udacity/deep-reinforcement-learning) to setup the environment.
     After setting up the environment, download the files in this repository and put them into /your path/deep-reinforcement-learning/p1_navigation
     Download the banana environment follow the instruction on [p1_navigation](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)
   - My solution codes
     utils folder: implementation of prioritized experience replay. This is directly cloned from [DeepRL tutorial](https://github.com/qfettes/DeepRL-Tutorials/tree/master/utils)
     
     | Code   | function description | hyperparameter involved |
     | ---    | -------------------- | ----------------------- |
     |model.py|Implement a dueling DQN network with 2 fully connected layers| number of neurons in the first and second fully connected layers, both set to 64 by default|
     |dqn_agent.py|Implement an agent based on the double DQN algorithm. The Q values used in this agent are estimated by the dueling netwrok in model.py. The replay buffer used in the agent is the priotized experience replay buffer in the utils folder.|α - determine how much priotization is used |
   
   - Hyperparameters selection

## Part 4: Demo for the trained agent
