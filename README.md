# RL-Project-Navigation
Train an agent by value-based reinforcement learning to navigate in a large world and collect yellow bananas, while avoiding blue bananas

## Part 0: Basic reinforcement learning concept involved
![](https://github.com/CenturyLiu/RL-Project-Navigation/blob/master/rl-basic.png)
> The agent-environment interaction in Reinforcement Learning.(Source: [Sutton and Barto, 2017](http://incompleteideas.net/book/RLbook2020.pdf))

In short, the simplest reinforcment learning framework involves an agent interacting with the environment. At each time step, the agent observes the **state** of the environment, choose valid **action** and receive **reward** from the environment. The interaction between the agent and the environment forms a sequence of s<sub>0</sub> ,a<sub>0</sub>, r<sub>1</sub> ,s<sub>1</sub> ,a<sub>1</sub>, r<sub>2</sub>,...,s<sub>T</sub> ,a<sub>T</sub>, r<sub>T+1</sub>       (s - states,a - action,r - reward). The goal of the agent is to maximize the total reward it can receive.

Based on whether the interation has well-defined end, reinforcement learning tasks can be divided in to episodic tasks and continuing tasks. Episodic tasks have well-defined begin and end point (eg. Drive a car from your position to destination). Continuing tasks will run forever (eg. The car is running on an endless road). This project is discussing my solution to an episodic task.

Mathematically, reinforcement learning can be described as a **Markov Decision Process (MDP)**, which involves **state space S**, **action space A** and reward 

## Part 1: Problem to solve

## Part 2: Idea for solving the problem

## Part 3: Project implementation
   - Prerequisite for code installation
   - My solution codes
   - Hyperparameters selection

## Part 4: Demo for the trained agent
