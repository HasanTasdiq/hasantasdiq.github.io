---
title: "QuRA: Reinforcement Learning–Based Routing for Quantum Networks"
date: 2024-06-01
summary: "A deep reinforcement learning–based routing framework that enables scalable, fidelity-aware quantum routing under dynamic traffic."
tags: ["Quantum Networks", "Reinforcement Learning", "Quantum Routing"]
---

## Project Overview

This project introduces **QuRA**, a **Deep Q-Reinforcement Learning (DQRL)**–based routing framework for quantum networks.  
Quantum routing is challenging due to **fidelity degradation**, **probabilistic links**, and **resource contention**. Existing shortest-path and ILP-based solutions either perform poorly or fail to scale.

QuRA addresses these issues by learning **adaptive, per-hop routing decisions** that dynamically balance fidelity, congestion, and resource usage, achieving high performance with significantly lower computation cost.

---

## Key Ideas

- **Fidelity-aware routing:** Explicitly accounts for fidelity loss from entanglement swapping  
- **Hop-by-hop decision making:** Reduces complexity while maintaining high routing quality  
- **Multi-request support:** Jointly schedules and routes concurrent requests  
- **Topology generalization:** Works across different networks without retraining

---

## QuRA Routing Framework

![QuRA routing framework overview](/projects/qura/fig1.png)

QuRA operates via a centralized controller with a DQRL agent that selects a **(request, next-hop)** action at each step based on the global network state, incrementally constructing end-to-end entangled paths.

---

## Fidelity-Aware Motivation

![Fidelity-aware routing example](/projects/qura/fig2.png)

Hop-count–based shortest paths may fail to meet fidelity thresholds, while QuRA learns to select **longer but higher-fidelity paths**, enabling successful entanglement delivery under strict quality constraints.

---

## Performance Highlights

![QuRA performance comparison](/projects/qura/fig3.png)

- Up to **90% higher success rate** under fidelity constraints  
- **~79% runtime reduction** compared to ILP-based routing  
- Scales efficiently with network size and traffic load  
- Adapts to dynamic conditions without retraining

---

## Publication

**Tasdiqul Islam**,  Engin Arslan,  Md Arifuzzaman 

*QuRA: Reinforcement Learning–Based Routing for Quantum Networks*  
IEEE Consumer Communications and Networking Conference (CCNC’26). IEEE, Las Vegas, NV, USA. (Accepted for publication)

---
