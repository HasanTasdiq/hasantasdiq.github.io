---
title: "RL Notes: Huggin Face RL Course"
date: 2025-10-10
tags: ["RL", "Neural Network", "Machine Learning", "AI", "Hugging Face"]
math: true
draft: false
---

# [HFRL Unit-1](https://huggingface.co/learn/deep-rl-course/en/unit1)

## Summary
* Reinforcement Learning is a method where an agent learns by interacting with its environment, using trial and error and feedback from rewards.

* The goal of an RL agent is to maximize its expected cumulative reward, based on the idea that all goals can be framed as maximizing this reward.

* The RL process is a loop that outputs a sequence of state, action, reward, and next state.

* Expected cumulative reward is calculated by discounting future rewards, giving more weight to immediate rewards since they are more predictable than long-term ones.

* To solve an RL problem, we find an optimal policy, the AI's "brain" that decides the best action for each state to maximize expected return.

## Two ways to find Optimal Policy

1. **Policy-based** methods directly optimize the policy by adjusting its parameters to maximize expected return.
2. **Value-based** methods train a value function that estimates the expected return for each state and use it to define the policy.

Deep RL is "Deep" because it uses deep neural networks to estimate the action to take (policy-based) or to estimate the value of a state (value-based).

# [HFRL Unit-2](https://huggingface.co/learn/deep-rl-course/en/unit2)
## Value-based methods
* Value-based methods estimate the value of each state (or state-action pair) and derive the policy from these values.
* We learn a value function that maps a state to the expected value of being at that state.
![Value Function](/blogs/tutorials/huggingfaceRL/vbm-1.jpg)


* The value of a state is the expected discounted return obtained by following the policy from that state.

### What does it mean to "follow a policy"?
- The goal of an RL agent is to have an optimal policy $\pi^*$ that maximizes the expected return from any state.
    - Policy-based methods $\rightarrow$ directly train the policy to select what action to take given a state $\rightarrow \pi(a|s) \rightarrow$ We do not need to learn a value function.
        - Policy takes the current state as input and outputs an action or a distribution over possible actions.
        - We don't define the policy explicitly; instead, we learn it through interactions with the environment.
        ![Policy-based](/blogs/tutorials/huggingfaceRL/two-approaches-2.jpg)
    - Value-based methods $\rightarrow$ trains the policy indirectly by learning a value function $\rightarrow V(s) \rightarrow$ We do not need to learn the policy explicitly.
        - The value function takes the current state as input and outputs the expected return (value) of that state.
        - The policy is not trained or learned directly $\rightarrow$ it is derived from the value function given specific rules.
        - For example, a common rule is to select the action that leads to the state with the highest value (greedy policy).
        ![Value-based](/blogs/tutorials/huggingfaceRL/two-approaches-3.jpg)
### The Difference between Value-based and Policy-based methods
- In policy-based methods, the policy is learned directly by training a neural network to output actions based on states.
    - Optimal policy $\pi^*$ is learned directly.
- In value-based methods, the policy is derived from a learned value function, which estimates the expected return of states.
    - Optimal policy $\pi^*$ is derived from the optimal value function $V^*(s)$ or the action-value function $Q^*(s, a)$.
    ![Link](/blogs/tutorials/huggingfaceRL/link-value-policy.jpg)


### Two types of Value Functions
1. **State-Value Function** $V_\pi(s)$: Estimates the expected return starting from state $s$ and following policy $\pi$.
    - It gives the value of being in a particular state.
    - Formula:
    $$V_\pi(s) = \mathbb{E}_\pi [G_t | S_t = s]$$
    - Where $G_t$ is the return (cumulative discounted reward) from time step $t$.
    ![State-Value Function](/blogs/tutorials/huggingfaceRL/state-value-function-1.jpg)
2. **Action-Value Function** $Q_\pi(s, a)$: Estimates the expected return starting from state $s$, taking action $a$, and following policy $\pi$ thereafter.
    - It gives the value of taking a particular action in a particular state.
    - Formula:
    $$Q_\pi(s, a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a]$$
    ![Action-Value Function](/blogs/tutorials/huggingfaceRL/action-state-value-function-1.jpg)

**Difference**: The state-value function evaluates the value of being in a state, while the action-value function evaluates the value of taking a specific action in a state.

**Both** case are used to derive the optimal policy $\pi^*$ by selecting actions that maximize the expected return.

**Problem**: To calculate the value of a state or state-action pair, we need to know the expected return $G_t$, which depends on future rewards and the policy $\pi$. This can be computationally expensive and complex, especially in environments with many states and actions. Bellman equations provide a recursive way to compute these values efficiently.


## Bellman Equations

- Simplifies the computation of value functions ($V_\pi(s)$ and $Q_\pi(s, a)$) by breaking them down into immediate rewards and the value of subsequent states.
- Instead of calculating the expected return for each state or each state-action pair, we can use the Bellman equation.
- The Bellman equation expresses a state’s value recursively as the immediate reward plus the discounted value of the next state: 
$$V(S_t) = R_{t+1} + \gamma V(S_{t+1})$$
![Bellman Equation](/blogs/tutorials/huggingfaceRL/bellman4.jpg)

- In summary, the Bellman equation simplifies value estimation by expressing it as the immediate reward plus the discounted value of the next state.

- Monte Carlo methods and Temporal Difference (TD) learning are two primary approaches to estimate value functions in reinforcement learning.


## Monte Carlo Methods
- Monte Carlo waits until the end of an episode to calculate the total return ($G_t$) and then updates the value function ($V_\pi(s)$ or $Q_\pi(s, a)$) based on this return.
- It requires complete episodes.
![Monte Carlo Methods](/blogs/tutorials/huggingfaceRL/monte-carlo-approach.jpg)
- The update rule for the state-value function is:
$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$
- Where $\alpha$ is the learning rate, and $G_t$ is the total return from time step $t$.
- Then restart the episode and repeat.
![Monte Carlo Update](/blogs/tutorials/huggingfaceRL/MC-3p.jpg)

**Example of Monte Carlo:**
- We initialize the value function $V(s)$ arbitrarily (e.g., all zeros) for all states $s$.
- Our learning rate $\alpha$ is set to 0.1 and discount factor $\gamma$ is 1 (for simplicity).
- We run an episode in the environment, collecting states, actions, and rewards until the episode ends.
- At the end of the episode, we calculate the return $G_t$ for each time step $t$.
- We update the value function for each state visited during the episode using the update rule.
- We repeat this process for many episodes, gradually improving our value function estimates.

## Temporal Difference (TD) Learning
- TD learning updates the value function ($V_\pi(s)$ or $Q_\pi(s, a)$) at each time step using the immediate reward and the estimated value of the next state.
- It does not require complete episodes and can learn from incomplete sequences.
- Because we didn't wait until the end of the episode, we don't have the full return $G_t$. 
    - Instead, we use the immediate reward $R_{t+1}$ and the estimated value of the next state $V(S_{t+1})$ to update our value function.
    - This is called bootstrapping because we are using our current estimate to improve itself incrementally.
![TD Learning](/blogs/tutorials/huggingfaceRL/TD-1.jpg)
- The update rule for the state-value function is:
$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$
- Where $\alpha$ is the learning rate, $R_{t+1}$ is the immediate reward, and $V(S_{t+1})$ is the estimated value of the next state.
- Then move to the next state and repeat.
- This method is known as **TD(0)** because it updates the value function based on a one-step lookahead.
- The estimated return is known as the TD target:
$$\text{TD Target} = R_{t+1} + \gamma V(S_{t+1})$$

![TD(0) Update](/blogs/tutorials/huggingfaceRL/TD-1p.jpg)

### Summary of Monte Carlo vs TD Learning
| Aspect               | Monte Carlo Methods                          | Temporal Difference (TD) Learning          |
|----------------------|----------------------------------------------|--------------------------------------------|
| Update Timing       | Updates value function at the end of an episode | Updates value function at each time step      |
| Requirement         | Requires complete episodes                    | Can learn from incomplete sequences         |
| Return Calculation  | Uses total return $G_t$                       | Uses immediate reward and estimated next state value |
| Bootstrapping       | No                                           | Yes                                        |


## On-policy vs Off-policy Learning
- **On-policy** methods learn the value of the policy being executed (the same policy used to make decisions).
    - Example: SARSA (State-Action-Reward-State-Action)
- **Off-policy** methods learn the value of a different policy than the one being executed (the behavior policy).
    - Example: Q-learning

## Q-learning
- Q-Learning is an off-policy value-based method that uses a TD approach to train its action-value function $Q(s, a)$.
- Approximate the optimal action-value function $Q^*(s, a)$, which gives the maximum expected return for taking action $a$ in state $s$ and following the optimal policy thereafter.
![Q-Learning](/blogs/tutorials/huggingfaceRL/Q-function.jpg)