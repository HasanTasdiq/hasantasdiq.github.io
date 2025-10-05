+++
title = "Evaluating Fuzzers on Cryptographic Protocols"
date = 2025-09-25
tags = ["Fuzzer", "Security", "Cryptography"]
draft = false
+++

# Evaluating Fuzzers on Cryptographic Protocols
_Paper accepted at SBFT 2025 Workshop (co-located with ICSE 2025)_

<!-- ![ScreenShot](workflow_csfuzz.png) -->
![Workflow](/projects/fuzzeval/workflow_csfuzz.png)


General-purpose fuzzers often struggle to identify vulnerabilities in cryptographic libraries because they cannot generate inputs that satisfy strict protocol validations. This study evaluates modern fuzzers on their ability to produce **context-sensitive inputs** for PKCS#1-v1.5 signature verification. Our findings show that **semantic awareness**—understanding complex relationships between input fields—is more critical than code coverage for testing these security-critical implementations.

**Paper:** [FuzzEval: Assessing Fuzzers on Generating Context-Sensitive Inputs](https://arxiv.org/abs/2409.12331)  

🏆 Submitted as a [short paper](/projects/fuzzeval/csfuzz-sbft2025.pdf), it **received the [Best Paper Award](/projects/fuzzeval/sbft2025-award.pdf)** at the [SBFT 2025 Workshop](https://sbft25.github.io/), co-located with ICSE 2025.

**BibTeX:**
```bibtex
@article{hasan2024fuzzeval,
  title={FuzzEval: Assessing Fuzzers on Generating Context-Sensitive Inputs},
  author={Hasan, S Mahmudul and Kozyreva, Polina and Hoque, Endadul},
  journal={arXiv preprint arXiv:2409.12331},
  year={2024}
}
```