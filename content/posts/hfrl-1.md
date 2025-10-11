---
title: "RL Notes: HFRL Unit-1"
date: 2025-10-10
tags: ["RNN", "Neural Network", "Machine Learning", "NLP"]
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
![Value Function](/static/blogs/tutorials/vbm-1.jpg)


* The value of a state is the expected discounted return obtained by following the policy from that state.

### What does it mean to "follow a policy"?
- The goal of an RL agent is to have an optimal policy $\pi^*$ that maximizes the expected return from any state.
    - Policy-based methods $\rightarrow$ directly train the policy to select what action to take given a state $\rightarrow \pi(a|s) \rightarrow$ We do not need to learn a value function.
        - Policy takes the current state as input and outputs an action or a distribution over possible actions.
        - We don't define the policy explicitly; instead, we learn it through interactions with the environment.
        ![Policy-based](/static/blogs/tutorials/two-approaches-2.jpg)
    - Value-based methods $\rightarrow$ trains the policy indirectly by learning a value function $\rightarrow V(s) \rightarrow$ We do not need to learn the policy explicitly.
        - The value function takes the current state as input and outputs the expected return (value) of that state.
        - The policy is not trained or learned directly $\rightarrow$ it is derived from the value function given specific rules.
        - For example, a common rule is to select the action that leads to the state with the highest value (greedy policy).
        ![Value-based](/static/blogs/tutorials/two-approaches-3.jpg)
### The Difference between Value-based and Policy-based methods
- In policy-based methods, the policy is learned directly by training a neural network to output actions based on states.
    - Optimal policy $\pi^*$ is learned directly.
- In value-based methods, the policy is derived from a learned value function, which estimates the expected return of states.
    - Optimal policy $\pi^*$ is derived from the optimal value function $V^*(s)$ or the action-value function $Q^*(s, a)$.
    ![Link](/static/blogs/tutorials/link-value-policy.jpg)