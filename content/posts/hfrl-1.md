---
title: "RL Notes: HFRL Unit-1"
date: 2025-10-10
tags: ["RNN", "Neural Network", "Machine Learning", "NLP"]
math: true
draft: true
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



