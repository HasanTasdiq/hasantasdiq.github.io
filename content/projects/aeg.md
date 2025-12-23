---
title: "Adaptive Entanglement Generation for Quantum Routing"
date: 2024-07-01
summary: "A deep reinforcement learning framework for fast and scalable entanglement generation in quantum networks."
tags: ["Quantum Networks", "Reinforcement Learning", "Quantum Routing"]
draf: False
---

## Project Overview

**Adaptive Entanglement Generation (AEG)** is a deep reinforcement learning–based framework that improves the scalability and efficiency of quantum network routing.  
It replaces slow integer linear programming (ILP)–based link selection with **Deep Q-Learning**, enabling real-time decisions while maintaining near-optimal performance.

In addition, AEG leverages the **temporal persistence of entanglement** through caching and proactive swapping to significantly improve request success rates.

---

## Key Ideas

- **RL-based link selection:** Predicts which links to entangle in each time slot, achieving up to **20× faster execution** than ILP.
- **Entanglement caching:** Reuses unused entangled links across time slots, improving throughput by **10–20%**.
- **Proactive entanglement swapping:** Extends entanglement on high-demand path segments ahead of time, yielding up to **107% improvement** over baseline routing methods.

---

## Why Link Selection Matters

![Impact of entanglement link selection](/projects/aeg/fig1.png)

*Choosing better links for entanglement generation directly increases the number of feasible end-to-end paths and routing reliability.*

---

## Learning-Based Entanglement Control

![RL-based link selection model](/projects/aeg/fig2.png)

The Deep Q-Learning model takes network topology, distances, and request demand as input and outputs a set of candidate links for entanglement generation in parallel.

---

## Performance Highlights

![Performance comparison](/projects/aeg/fig4.png)

- Up to **107% higher request success rate** compared to random and shortest-path methods  
- **Near-constant runtime** as network load increases  
- Comparable performance to ILP with **orders-of-magnitude lower latency**

---

**Preprint:** [Adaptive Entanglement Generation for Quantum Routing](https://arxiv.org/abs/2505.08958)  


**BibTeX:**
```bibtex
@article{islam2025adaptive,
  title={Adaptive Entanglement Generation for Quantum Routing},
  author={Islam, Tasdiqul and Arifuzzaman, Md and Arslan, Engin},
  journal={arXiv preprint arXiv:2505.08958},
  year={2025}
}
```
