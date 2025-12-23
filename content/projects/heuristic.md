---
title: "A Heuristic Approach for Scalable Quantum Repeater Deployment Modeling"
date: 2023-10-01
summary: "Scalable heuristic algorithms for near-optimal quantum repeater placement with orders-of-magnitude speedup over ILP."
tags: ["Quantum Networks", "Quantum Repeaters", "Heuristic Optimization"]
---

## Project Overview

This project proposes **fast heuristic methods** for deploying quantum repeaters in large-scale quantum networks.  
While Integer Linear Programming (ILP) can find optimal placements, it does not scale. The proposed heuristics achieve **near-optimal solutions** while reducing computation time from **days to seconds**.

---

## Key Ideas

- **Coverage-driven placement** using a maximum entanglement distance \( L_{max} \)  
- **Heuristic optimization** instead of exponential-time ILP  
- **Scalable and failure-aware** deployment for real-world networks

---

## Heuristic Approaches

![Quantum repeater deployment overview](/projects/heuristic/fig1.png)

Two complementary strategies are introduced:

- **Multi-Center Approach (MCA):** Selects multiple coverage centers and adds intermediate repeaters only when needed  
- **Single Center Approach (SCA):** Gradually expands coverage from one center, jointly ensuring coverage and connectivity

---

## Performance Highlights

![Heuristic vs ILP comparison](/projects/heuristic/fig2.png)

- Near-optimal repeater count (often matching ILP)  
- Execution time reduced by **several orders of magnitude**  
- Validated on **SURFnet** and **ESnet** topologies  
- Extended to tolerate **node and link failures**

---

## Publication

**Tasdiqul Islam**, Engin Arslan  
[A Heuristic Approach for Scalable Quantum Repeater Deployment Modeling](https://ieeexplore.ieee.org/abstract/document/10223375) 
IEEE Conference on Local Computer Networks (LCN), 2023

**BibTeX:**
```bibtex
@inproceedings{islam2023heuristic,
  title={A heuristic approach for scalable quantum repeater deployment modeling},
  author={Islam, Tasdiqul and Arslan, Engin},
  booktitle={2023 IEEE 48th Conference on Local Computer Networks (LCN)},
  pages={1--9},
  year={2023},
  organization={IEEE}
}
```