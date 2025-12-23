---
title: "Proactive Entanglement Swapping for Quantum Networks"
date: 2024-06-01
summary: "A reinforcement learning framework that proactively performs entanglement swapping to improve throughput and reduce routing overhead in quantum networks."
tags: ["Quantum Networks", "Reinforcement Learning", "Entanglement Swapping"]
---

## Project Overview

This project introduces a **reinforcement learning–based proactive entanglement swapping (PES)** framework for quantum networks.  
Instead of performing entanglement swapping only after routing requests arrive, the proposed approach **anticipates future demand** and extends entanglement along frequently used path segments in advance.

By reducing the effective path length and swapping operations during routing, PES significantly improves network throughput and scalability.

---

## Key Ideas

- **Proactive swapping:** Performs entanglement swapping ahead of time on high-demand segments  
- **RL-based decision making:** Learns which nodes should perform swapping based on network state  
- **Reduced routing overhead:** Shorter paths and fewer swaps during request servicing

---

## Proactive vs Reactive Swapping

![Proactive entanglement swapping illustration](/projects/pes/fig1.png)

*Proactive swapping creates longer entangled links before requests arrive, enabling faster and more reliable routing.*

---

## Learning Framework

![RL-based proactive swapping model](/projects/pes/fig2.png)

The RL agent observes network topology, current entanglement states, and request statistics, and selects nodes where swapping should be performed proactively to maximize long-term throughput.

---

## Performance Highlights

![Performance comparison](/projects/pes/fig3.png)

- Up to **100%+ improvement** in request success rate over random and shortest-path baselines  
- Lower routing latency due to reduced swapping during path construction  
- Consistent performance gains under high traffic loads

---

## Publication

**Tasdiqul Islam**, Md Arifuzzaman, Engin Arslan  
[Reinforcement Learning–Based Proactive Entanglement Swapping for Quantum Networks](https://ieeexplore.ieee.org/abstract/document/10628215)  
IEEE QCNC 2024

**BibTeX:**
```bibtex
@inproceedings{islam2024reinforcement,
  title={Reinforcement Learning Based Proactive Entanglement Swapping for Quantum Networks},
  author={Islam, Tasdiqul and Arifuzzaman, Md and Arslan, Engin},
  booktitle={2024 International Conference on Quantum Communications, Networking, and Computing (QCNC)},
  pages={135--142},
  year={2024},
  organization={IEEE}
}
```

