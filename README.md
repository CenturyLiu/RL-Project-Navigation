# RL-Project-Navigation
Train an agent by value-based reinforcement learning to navigate in a large world and collect yellow bananas, while avoiding blue bananas

## Part 0: Basic reinforcement learning concept involved
![](https://github.com/CenturyLiu/RL-Project-Navigation/blob/master/rl-basic.png)
> The agent-environment interaction in Reinforcement Learning.(Source: [Sutton and Barto, 2017](http://incompleteideas.net/book/RLbook2020.pdf))

In short, the simplest reinforcment learning framework involves an agent interacting with the environment. At each time step, the agent observes the **state** of the environment, choose valid **action** and receive **reward** from the environment. The interaction between the agent and the environment forms a sequence of S<sub>0</sub> ,A<sub>0</sub>, R<sub>1</sub> ,S<sub>1</sub> ,A<sub>1</sub>, R<sub>2</sub>,...,S<sub>T</sub> ,A<sub>T</sub>, R<sub>T+1</sub>       (s - states,a - action,r - reward). The goal of the agent is to learn to choose action to maximize the total reward it can receive. The total reward at time step t is given by G<sub>t</sub> = R<sub>t+1</sub> + R<sub>t+2</sub> + R<sub>t+3</sub> + ... Intuitively, agent should give more emphasis on getting reward in the near future, and pay less attention to the reward in the far future, we introduce a **discount rate γ (gamma)**. The total reward with disount rate is calculated by G<sub>t</sub> = γ\*R<sub>t+1</sub> + γ\*R<sub>t+2</sub> + γ\*R<sub>t+3</sub> + ...

Based on whether the interation has well-defined end, reinforcement learning tasks can be divided in to episodic tasks and continuing tasks. Episodic tasks have well-defined begin and end point (eg. Drive a car from your position to destination). Continuing tasks will run forever (eg. The car is running on an endless road). This readme is discussing my solution to an episodic task.

Mathematically, reinforcement learning can be described as a **Markov Decision Process (MDP)**, which involves **state space S**, **action space A** and **reward R**. For more detailed description of this process, I recommend you take a course for reinforcement learning (eg: [Udacity deep reinforcement learning nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)) or refer to the book [Sutton and Barto, 2017](http://incompleteideas.net/book/RLbook2020.pdf).  

## Part 1: Problem to solve

## Part 2: Idea for solving the problem

## Part 3: Project implementation
   - Prerequisite for code installation
   - My solution codes
   - Hyperparameters selection

## Part 4: Demo for the trained agent
