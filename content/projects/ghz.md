---
title: "Quantum Key Distribution with Single-Qubit Transmission"
date: 2024-06-01
summary: "A GHZ-state–based quantum key distribution protocol that transmits multiple key bits using a single qubit and quantum nondemolition measurements."
tags: ["Quantum Cryptography", "QKD", "GHZ States"]
---

## Overview

This project proposes a **GHZ-state–based Quantum Key Distribution (QKD)** scheme that significantly reduces qubit transmission overhead.  
By generating an N-qubit GHZ state and transmitting only **one qubit**, the protocol enables the secure transfer of **multiple classical key bits**.

The approach combines **quantum teleportation** and **quantum nondemolition (QND) measurements**, making it well-suited for capacity-limited quantum networks.

---

## Key Contributions

- **Single-qubit transmission:** Sends one qubit per key bit instead of one qubit per bit per protocol round  
- **GHZ-based encoding:** Transfers up to N-1 classical bits using an N-qubit GHZ state  
- **High efficiency:** Achieves high efficiency outperforming BB84, E91, and related schemes

---

## Protocol Concept

![GHZ-based QKD protocol](/projects/ghz/fig1.png)

*Alice generates a multi-qubit GHZ state, transmits one qubit to Bob, and encodes key bits using teleportation and QND measurements.*

---

## Results

- Successfully simulated GHZ-based QKD using **NetSquid**
- Demonstrated higher key-to-qubit efficiency than classical QKD protocols
- Feasible with experimentally demonstrated large-scale GHZ states

---

## Publication

**Tasdiqul Islam**, Engin Arslan  
[Quantum Key Distribution with Single Qubit Transmission](https://ieeexplore.ieee.org/abstract/document/10628318)
(Poster) IEEE QCNC 2024

**BibTeX:**
```bibtex
@inproceedings{islam2024quantum,
  title={Quantum Key Distribution with Single Qubit Transmission},
  author={Islam, Tasdiqul and Arslan, Engin},
  booktitle={2024 International Conference on Quantum Communications, Networking, and Computing (QCNC)},
  pages={357--358},
  year={2024},
  organization={IEEE}
}

```

