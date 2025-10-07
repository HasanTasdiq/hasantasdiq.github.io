---
title: "RWKV: The Revolutionary Architecture Bridging RNNs and Transformers"
date: 2025-09-30
tags: ["Transformer", "Neural Network", "Machine Learning", "NLP"]
math: true
draft: true
---

# RWKV: The Revolutionary Architecture Bridging RNNs and Transformers

*How RWKV combines the best of both worlds to achieve linear complexity without sacrificing performance*

---

## Part I — Foundations

### Why RWKV Matters

The **Receptance Weighted Key Value (RWKV)** architecture rethinks sequence modeling.
It fuses the **parallel-training power of Transformers** with the **streaming efficiency of RNNs**--solving a long-standing trade-off:

| Architecture    | Training          | Inference   | Long Sequences |
| --------------- | ----------------- | ----------- | -------------- |
| **RNN / LSTM**  | Sequential (slow) | Fast $\mathcal{O}(T)$  | Stable         |
| **Transformer** | Parallel (fast)   | Slow $\mathcal{O}(T^2)$ | Memory-heavy   |
| **RWKV**        | Parallel (fast)   | Fast $\mathcal{O}(T)$  | Excellent      |

RWKV can train in parallel like a Transformer but infer step-by-step like an RNN, while retaining Transformer-level quality on large corpora.

---

### The Fundamental Insight

Instead of computing *pairwise attention scores* between every pair of tokens, RWKV keeps a **compact state** summarizing the past via exponentially decayed key--value statistics.

Think of it as turning the attention matrix  $$A_{ij}=\mathrm{softmax}(Q_iK_j^\top)V_j$$, into a *recurrent update*: 
$$state_t = \lambda \cdot state_{t-1} + f(K_t,V_t)$$ where $\lambda$ is a learned decay.

This simple shift brings **linear complexity** and **constant-memory inference**.

---

### Anatomy of RWKV (RWKV-4)

RWKV uses two cooperating sub-blocks per layer:

1. **Time-Mixing** -- captures relationships *across time* (sequence order).
   Analogous to the Transformer's **self-attention**.

2. **Channel-Mixing** -- processes information *within* each token's embedding.
   Analogous to the Transformer's **feed-forward network (FFN)**.

---

### The Four Fundamental Symbols

| Symbol                 | Meaning                                                 | Analogy                             |
| ---------------------- | ------------------------------------------------------- | ----------------------------------- |
| **R (Receptance)**     | Gate deciding how much of the aggregated context to use | LSTM's input gate / attention query |
| **W (Weight / Decay)** | Exponential time-decay controlling memory span          | Positional bias / forget gate       |
| **K (Key)**            | Determines how much each past token contributes         | Transformer key                     |
| **V (Value)**          | Information carried forward                             | Transformer value                   |

---

### Time-Mixing — The Heart of RWKV-4

For token $t$, with embedding $x_t$ and previous token embedding $x_{t-1}$, we compute:

$$
r_t = W_r(\mu_r\odot x_t + (1-\mu_r)\odot x_{t-1})\\
$$
$$
k_t = W_k(\mu_k\odot x_t + (1-\mu_k)\odot x_{t-1})\\
\tag{Eq 1--3}
$$
$$
v_t = W_v(\mu_v\odot x_t + (1-\mu_v)\odot x_{t-1})\\
$$

- The learned vector $\mu$ mixes current and previous token -- **a smooth token shift** improving gradient flow. Size: $(d,)$ where d is the embedding dimension.
- ($W_r,W_k,W_v$) are projection matrices, i.e. linear layers in $\texttt{PyTorch}$ with bias disabled.

#### Transformer analogy

This blending plays the role of **query/key/value projection plus positional encoding** in attention layers -- but implemented as a local interpolation instead of explicit position embeddings.

#### RNN analogy

Equivalent to feeding the previous hidden state into the next computation, but through a **learned linear interpolation** rather than additive recurrence.

---

### Weighted Key--Value Aggregation

RWKV replaces the attention softmax with a **decayed weighted average**:

$$
wkv_t =
\frac{
\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i}\odot v_i + e^{u+k_t}\odot v_t
}{
\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}
}
\tag{Eq 4}
$$

* $w$ -- learned decay (per channel)
* $u$ -- bias emphasizing the current token
* $e^{k_i}$ — importance weight

Essentially, each channel maintains its own **softmax-like moving average** over time.

#### Example (Intuition)

Imagine processing a sentence:

> “The cat sat on the mat.”

Older words fade with ($e^{-w \Delta t}$).
If $w = 0.2$, a word 5 steps ago keeps $e^{-1}=0.37$ of its influence.

#### Transformer comparison

Softmax attention computes explicit pairwise weights: $\mathcal{O}(T^2)$.
RWKV's exponential decay acts like **continuous-time softmax attention**, but updated recursively -- $\mathcal{O}(T)$.

#### RNN comparison

An RNN's hidden state mixes past info via nonlinear recurrent matrices.
RWKV's accumulation is *linear in $v$* and controlled via *learned decays*, giving more stable gradients.

---

### Receptance and Output Projection
$$
o_t = W_o\big(\sigma(r_t) \odot wkv_t\big)
\tag{Eq 5}
$$

* ($\sigma(r_t)$) is a sigmoid gate (0--1) deciding how much contextual info to output.
* ($W_o$) projects back to the hidden dimension.

#### Transformer analogy

Comparable to attention's output projection ($W_O(QK^\top V)$), but gating occurs *per channel* rather than *per token pair*.

#### RNN analogy

Like the LSTM's **output gate**, but data-driven through *$r_t$* rather than hidden-state recursion.

---

### Channel-Mixing — Intra-Token Processing

After time-mixing, each token's features are refined:

$$
r'_t = W'_r\left((\mu'_r \odot x_t) + (1 - \mu'_r) \odot x_{t-1}\right)
$$
$$
k'_t = W'_k\left((\mu'_k \odot x_t) + (1 - \mu'_k) \odot x_{t-1}\right)
\tag{Eq 6 -- 8}
$$
$$
o'_t = \sigma(r'_t) \odot \big(W'_v(\max(k'_t,0))^2\big)
$$

* Squared ReLU $((\max(k'_t,0))^2)$ acts like a lightweight MLP non-linearity.
* Receptance again gates the output.

#### Transformer analogy

Direct counterpart of the **FFN (two linear layers + GELU)** inside each Transformer block.

#### RNN analogy

Acts like the **nonlinear state transformation** between recurrent steps, separated from temporal recursion.

---

### Mini Numeric Example

Let's collapse everything to 1-D for clarity.

* ($x_1$ = 1.0, $x_2$ = 2.0)
* ($\mu$ = 0.6, $W$ = 1, $w$ = 0.3, $u$ = 0.0)
* ($k_1$ = 0.1, $k_2$ = 0.2, $v_1$ = 0.5, $v_2$ = 0.6)
Compute for $t$ = 2:

$$
r_2 = \mu x_2+(1-\mu)x_1 = 1.6\
$$

$$
\text{numerator} = e^{k_1}v_1 + e^{u+k_2}v_2
= 1.105(0.5)+1.221(0.6)=1.285\
$$

$$
\text{denominator} = e^{k_1}+e^{u+k_2}=2.326\
$$

$$
wkv_2 = 1.285/2.326=0.553
$$

Then ($o_2 = \sigma(r_2),wkv_2\approx0.832 × 0.553 \approx 0.46$).

*Interpretation:* the model blends the previous token's value (weighted 0.55) and the current one (0.73) -- a smooth, exponentially decayed attention in 1-D form.

---

### Complexity Check

| Model       | Time Complexity | Memory per token     | Parallel Training |
| ----------- | --------------- | -------------------- | ----------------- |
| Transformer | $\mathcal{O} (T^2)$          | $\mathcal{O} (T^2)$               | ✅                 |
| RWKV        | $\mathcal{O} (T)$           | $\mathcal{O} (1)$ (at inference) | ✅ (prefill)       |
| RNN         | $\mathcal{O} (T)$           | $\mathcal{O} (1)$                | ❌ (sequential)    |

RWKV combines the best of both worlds: *parallelizable training*, *streaming inference*, and *long-context retention*.

---

# 🦅 Part II -- RWKV-5 (Eagle)
_Multi-Head Matrix States and Expressive Decay_

RWKV-5, code-named **Eagle**, builds directly upon RWKV-4's success.
Its design question:

> “Can we give RWKV the representational power of *multi-head attention*
> without losing its RNN-like efficiency?”

---

## Why RWKV-5 Was Born

RWKV-4's channel-wise scalar state works well but can only store **one mixture of history** per feature dimension.
Transformers, in contrast, use **multiple heads**, each learning a different pattern of attention (syntax, semantics, position, etc.).

So Eagle extends RWKV by introducing:

* **Matrix-valued states** instead of single vectors
* **Per-head decays**
* **Smarter token-mix interpolation (`lerp`)**
* **Stability tricks** (contraction-bounded decays, SiLU, GroupNorm)

The result:
A model that behaves *mathematically like a recurrent Transformer with multi-head attention*, but runs in linear time.

---

## Linear Interpolation (`lerp`) – Token Shift Revisited

RWKV-4 used
\[
\mu_\square\odot x_t + (1-\mu_\square)\odot x_{t-1}.
\]

Eagle rewrites this as a cleaner **linear interpolation**:

\[
\operatorname{lerp}_\square(a,b)
= a + (b-a)\odot\mu_\square
\tag{Eq 9}
\]

where each \(\mu_\square\) is a learnable vector in $\mathbb{R}^D$.

So for every projection (r, k, v, g):

\[
\square_t = \operatorname{lerp}_\square(x_t, x_{t-1}) W_\square.
\tag{Eq 10}
\]

---

### Transformer Analogy

In Transformers, *relative position encodings* and *rotary embeddings* softly mix information from adjacent tokens.
`lerp` serves a similar role, but it's **learned** and **per-channel**, allowing continuous token shifts--no sinusoidal tables.

### RNN Analogy

This interpolation behaves like a *skip connection* from $x_{t-1}$ into $x_t$, providing the same gradient-stabilizing effect as residual recurrences in gated RNNs.

---

## Matrix-Valued Key–Value Accumulation

Eagle's key equation:
$$w = e^{-e^{\omega}}, \qquad 0 < w < 1$$


$$
\mathcal{WKV}_t = \operatorname{diag}(u) \, k_t^{\top} v_t + \sum_{i=1}^{t-1} \operatorname{diag}(w)^{t-1-i} \, k_i^{\top} v_i
\tag{Eq 11}
$$

Each head h maintains its own w vector and u bias.

**Interpretation**

* ($k_t^{\top}v_t$) → outer product forming a small matrix ($D_h × D_h$).
* The geometric decay ($\operatorname{diag}(w)^{t-1-i}$) discounts older contributions.
* The sum therefore stores *multiple decayed correlation patterns* -- exactly what attention heads learn in Transformers.

---

### Key Benefits

1. **Multi-Head Expressivity**
   Each head's matrix state models distinct dependencies -- akin to multi-head attention.

2. **Guaranteed Stability**
   The nested exp ensures $0 < w < 1$, preventing exploding/vanishing states.

3. **Constant-Time Update**
   The recurrence
   \[
   S_t = \operatorname{diag}(u)S_{t-1} + k_t^{\top}v_t
   \]
   allows $\mathcal{O}(1)$ state updates per token.

---

### Transformer Comparison

| Transformer (Self-Attention)                               | RWKV-5 (Eagle)                                         |
| ---------------------------------------------------------- | --------------------------------------------------     |
| Computes $QK^\top$ for every token pair $\mathcal{O}(T^2)$ | Maintains decayed sum of $K^\top V$ ($\mathcal{O}(T)$) |
| Attention weights via softmax (normalized across sequence) | Exponential decay per channel (normalized locally)     |
| Multi-head parallel attention matrices                     | Multi-head matrix states                               |
| Requires full context window to compute                    | Uses recurrent update (state only)                     |

---

### RNN Comparison

RNNs store a *single hidden vector* $h_t$.
Eagle's $\mathcal{WKV}_t$ acts as a structured, richer hidden state -- effectively a *learned covariance* of past inputs.

---

## Output and Gating

Eagle's output step refines the RWKV-4 formulation:

\[
o_t =
\operatorname{concat}\big(
\operatorname{SiLU}(g_t)
\odot
\operatorname{LayerNorm}(r_t ,\mathcal{WKV}_t)
\big)W_o
\tag{Eq 12}
\]

**Changes vs RWKV-4**

| RWKV-4               | RWKV-5 (Eagle)                               |
| -------------------- | -------------------------------------------- |
| Sigmoid gate         | SiLU (smoother, differentiable through zero) |
| Single vector state  | Matrix state per head                        |
| Shared LayerNorm     | Head-wise GroupNorm / LayerNorm              |
| $σ(r_t)$ gating only | Separate learned gate ($g_t$) for each head    |

These tweaks greatly improve numerical stability in deep stacks (> 24 layers).

---

### Concrete Mini Example

Assume two heads, head size $D_h = 2$.

At $t = 3$:

```text
Head 1:
  k₁ = [0.2, 0.1], v₁ = [0.5, 0.7]
  k₂ = [0.3, 0.1], v₂ = [0.6, 0.9]

Compute:
  k₁ᵀv₁ = [[0.10, 0.14],
             [0.05, 0.07]]
  k₂ᵀv₂ = [[0.18, 0.27],
             [0.06, 0.09]]
  decay w = 0.8

Accumulated matrix:
  S₃ = 0.8·S₂ + k₃ᵀv₃ ≈ 2×2 matrix per head.
```

Each head maintains its own $2 \times 2$ matrix capturing different patterns (e.g., entity vs relation).

---

## How Eagle Bridges to Transformers

### Attention Viewpoint

In a Transformer, for head $h$:
\[
\text{context}_t^{(h)} =
\sum_i \frac{e^{q_t^{(h)}·k_i^{(h)}/\sqrt{d_h}}}{Z_t},v_i^{(h)}.
\]

Eagle approximates this by replacing the softmax kernel with an **exponential decay kernel** and accumulating $k_i^{\top}v_i$ over time.
Hence the matrix state $\approx$ low-rank approximation to the attention Gram matrix.

### Efficiency

Because Eagle's recurrence doesn't require quadratic storage, it yields **$10--30\times$ less memory** during inference than Transformers of equal hidden size.

---

## Training & Prefill

Like RWKV-4, Eagle allows parallel "prefill":

during training you compute per-token r,k,v,g in parallel, then simulate the recurrence via efficient kernel operations (cumulative sums in log-space).
This keeps training speed comparable to Transformers.

---

## Summary Table – RWKV-4 vs RWKV-5

| Feature        | RWKV-4             | RWKV-5 (Eagle)            |
| -------------- | ------------------ | ------------------------- |
| State type     | Vector per channel | Matrix per head           |
| Decay          | Scalar per channel | Vector per head           |
| Token mixing   | $\mu$-based shift  | `lerp` function           |
| Activation     | Sigmoid            | SiLU + Gate $g_t$        |
| Normalization  | LayerNorm          | GroupNorm / HeadNorm      |
| Complexity     | $\mathcal{O}(T D)$ | $\mathcal{O}(T D)$ (same order)      |
| Expressiveness | Moderate           | High (multivariate state) |

---

## How Eagle Differs from Multi-Head Attention Mechanically

| Aspect                | Transformer Multi-Head     | RWKV-5 Eagle                            |
| --------------------- | -------------------------- | --------------------------------------- |
| *Computation*         | $(QK^\top V)$ per token pair | Cumulative $(K^\top V)$ state update      |
| *Cost*                | $\mathcal{O} (T^2 d)$                   | $\mathcal{O} (T d)$                                 |
| *Normalization*       | Softmax over tokens        | Exponential decay per step              |
| *Context window*      | Fixed (T)                  | Potentially infinite (while state fits) |
| *Memory size*         | $T \times d$                      | Constant per head                       |
| *Parallel training*   | Yes                        | Yes                                     |
| *Streaming inference* | No                         | Yes                                     |


---

## Take-Away

Eagle elevates RWKV from *a linear-time Transformer* to a *multi-head recurrent Transformer*.
It captures diverse dependencies like attention while remaining constant-memory at inference--an enormous win for edge and streaming models.

---


# 🪶 Part III — RWKV-6 (Finch) and RWKV-7 (Goose)

RWKV-6 (*Finch*) and RWKV-7 (*Goose*) are the most advanced forms of the RWKV family described in the survey  .
They introduce **data-dependent dynamics** and **generalized state updates**, pushing RWKV from “efficient Transformer alternative” to “dynamic memory system.”

---

## 🪶 RWKV-6 (Finch): Data-Dependent Dynamics

### 1️⃣ Problem Motivation

Eagle’s decays (w) and interpolations (\mu) are **fixed learned parameters**—identical for every token.
In long or context-rich sequences, the model needs **adaptive memory**: sometimes remember longer, sometimes forget faster.

Finch introduces *data-dependent lerp* (**ddlerp**) using **LoRA-style low-rank adapters**.

---

### 2️⃣ Data-Dependent Interpolation (Equation 18)

[
\begin{aligned}
\operatorname{lora}*\square(x)
&= \lambda*\square + \tanh(xA_\square)B_\square,\
\operatorname{ddlerp}*\square(a,b)
&= a + (b-a)\odot \operatorname{lora}*\square(a + (b-a)\odot\mu_\square).
\end{aligned}
\tag{Eq 13–14}
]

* (A_\square,B_\square) are low-rank LoRA matrices.
* (\lambda_\square) is a learned bias ensuring output ≈ 1 when input is small.
* The **tanh** introduces nonlinearity so interpolation changes *with content*.

So for each projection (r,k,v,g):

[
\square_t = \operatorname{ddlerp}*\square(x_t,x*{t-1})W_\square.
\tag{Eq 15}
]

---

### ⚖️ Transformer Comparison

In Transformers, attention weights vary with input content ((q_tk_i)).
RWKV-5’s fixed decays couldn’t; Finch’s ddlerp adds this **content sensitivity**, making decay and mixing depend on the current token—like **dynamic attention without explicit QK scores**.

### 🔁 RNN Comparison

This is analogous to an LSTM’s **input gate = σ(Wxₜ + Uhₜ₋₁)**—adaptive based on current input and state—but Finch achieves it linearly with LoRA adapters, keeping inference cheap.

---

### 3️⃣ Dynamic Decay (Equation 19)

Finch replaces the constant w with a dynamic vector wₜ computed from data:

[
\begin{aligned}
d_t &= \operatorname{lora}_d(\operatorname{ddlerp}*d(x_t,x*{t-1})),\
w_t &= \exp(-\exp(d_t)),[4pt]
\mathcal{WKV}_t
&= \operatorname{diag}(u)k_t^\top v_t

* \sum_{i=1}^{t-1}
  \operatorname{diag}!\Big(!\sum_{j=i+1}^{t-1}w_j!\Big)
  k_i^\top v_i.
  \end{aligned}
  \tag{Eq 16}
  ]

Now the decay rate itself *depends on the token sequence*.

---

### 💡 Intuition

Imagine reading:

> “John went to the park. He met Sarah. They played together.”

Finch can adapt:

* High decay for filler tokens (“to the park”)
* Low decay for entity tokens (“John”, “He”, “Sarah”)
  → maintains relevant entities longer.

---

### 🔢 Mini Example

Let the model output:
[
w_1=0.8,; w_2=0.95,; w_3=0.6.
]
At t = 4, earlier tokens get cumulative decay:
[
\text{weight(token 1)}=0.8\times0.95\times0.6=0.456.
]
So token 1 retains 45 % of its effect—stronger than a static global 0.6 decay.

---

### 🧮 Practical Result

Finch learns *when* to remember and *when* to forget, enabling better long-context recall (e.g., PG-19 and BookSum datasets).
Empirically, the survey reports ≈ 10 % improvement in long-context F1 over Eagle .

---

## 🪶 RWKV-7 (Goose): Advanced State Management

Goose overhauls the recurrence itself.
Its question:

> “Can we teach RWKV to **edit its own memory**, not just decay it?”

It answers with the **generalized delta rule**, bringing selective removal, insertion, and per-dimension learning rates.

---

### 1️⃣ Core Equation (Equation 20)

[
S_t
= S_{t-1}!\left(
\operatorname{diag}(w_t)
- \hat{\kappa}_t^\top(a_t\odot \hat{\kappa}_t)
\right)

* v_t^\top \tilde{\kappa}_t.
  \tag{Eq 17}
  ]

where

* (S_t): matrix state per head,
* (w_t): vector decay (0–1 per channel),
* (a_t): in-context learning rate (0–1 per channel),
* (\hat{\kappa}_t): *removal* key,
* (\tilde{\kappa}_t): *replacement* key,
* (v_t): value vector.

---

### 🧩 Term-by-Term Meaning

| Term                                                 | Role                         | Analogy                   |
| ---------------------------------------------------- | ---------------------------- | ------------------------- |
| (S_{t-1}\operatorname{diag}(w_t))                    | Decay of previous memory     | RNN forget gate           |
| (S_{t-1}\hat{\kappa}_t^\top(a_t\odot\hat{\kappa}_t)) | Targeted *removal* of memory | Attention-based erasure   |
| (v_t^\top\tilde{\kappa}_t)                           | Insert new info              | Key-Value write operation |

Thus Goose performs a full **erase-and-write** per token.

---

### 2️⃣ Generating Parameters (Equation 21)

[
\begin{aligned}
a_t &= \sigma(\operatorname{loramlp}_a(x_t)),\
w_t &= \exp(-e^{-0.5,\sigma(d_t)}),\
\hat{\kappa}_t &= k_t\odot\xi,\
\tilde{\kappa}_t &= k_t\odot(1-a_t\odot\alpha).
\end{aligned}
\tag{Eq 18}
]

* (\operatorname{loramlp}): small LoRA-MLP generating dynamic gates.
* (\xi,\alpha): learned scaling constants.

This gives per-token, per-channel control of memory editing.

---

### 3️⃣ Reading and Output (Equation 23)

[
\begin{aligned}
wkv_t &= wkv_{t-1}G_t + v_t^\top\tilde{\kappa}_t,\
u_t &= (r_t!(\rho!\odot!\tilde{\kappa}_t)^\top)v_t,\
p_t &= \operatorname{LayerNorm}(r_t wkv_t^\top) + u_t,\
o_t &= (g_t!\odot!p_t)W_o.
\end{aligned}
\tag{Eq 19}
]

Goose thus separates:

* **memory update** ((wkv_t)),
* **contextual read** ((r_t)),
* **output gate** ((g_t)).

---

### ⚖️ Transformer Comparison

| Transformer                        | RWKV-7 Goose                             |
| ---------------------------------- | ---------------------------------------- |
| Attention = weighted sum of values | State update = erase + add per dimension |
| KV-cache must store all tokens     | Compact matrix state                     |
| No explicit removal mechanism      | Explicit remove/add keys                 |
| One learning rate (global)         | Per-dimension adaptive rate (a_t)        |

So Goose behaves like a **memory-editing Transformer**—efficient and expressive.

---

### 🔁 RNN Comparison

| LSTM Gate      | Goose Equivalent           |
| -------------- | -------------------------- |
| Forget gate fₜ | (w_t)                      |
| Input gate iₜ  | (a_t)                      |
| Cell update    | (v_t^\top\tilde{\kappa}_t) |
| Output gate oₜ | (g_t)                      |

But Goose’s formulation is **matrix-valued and content-adaptive**, far richer than scalar LSTM gates.

---

### 🧠 Intuitive Story Example

“John entered the room. He saw Mary.”

| Token       | Goose Behavior                                          |
| ----------- | ------------------------------------------------------- |
| **John**    | create memory entry (entities = {John})                 |
| **entered** | add action state; low decay                             |
| **room**    | add location; decay moderate                            |
| **He**      | recall ‘John’; reinforce entity link                    |
| **Mary**    | new entity → removal of focus on room, add ‘Mary’ state |

Through (a_t,w_t,\hat{\kappa}_t,\tilde{\kappa}_t), Goose explicitly removes and inserts contextual traces—something Transformers only approximate via attention redistribution.

---

### 4️⃣ Mathematical Perspective

Goose’s update resembles the **delta rule** from adaptive filtering:
[
S_t = S_{t-1} + \eta_t (v_t^\top k_t - S_{t-1}).
]
But instead of a scalar ηₜ, Goose uses vector (a_t) and (w_t), making it a **learnable stochastic update rule** inside a neural net.

---

### 5️⃣ Computational View

| Operation                       | Cost                                 |
| ------------------------------- | ------------------------------------ |
| State update ((S_{t-1}\to S_t)) | O (Dₕ²) per head (but tiny matrices) |
| Read / write projections        | O (D)                                |
| Overall inference complexity    | O (T D) (total, linear in sequence)  |

Goose adds ≈ 5–10 % compute over Eagle but brings significant accuracy gains on reasoning and long-context tasks .

---

### 6️⃣ Why Goose Matters

1. **Dynamic Memory Editing** → explicit forgetting and replacement.
2. **Per-Feature Learning Rates** → better gradient conditioning.
3. **Expressive Theoretical Power** → shown to simulate non-trivial automata classes (NC¹ tasks).
4. **Long-Context Scaling** → handles 35 k + tokens without KV cache explosion.

---

### 🧮 Empirical Summary

| Benchmark           | RWKV-5 (Eagle) | RWKV-6 (Finch) | RWKV-7 (Goose) |
| ------------------- | -------------- | -------------- | -------------- |
| PG-19 (long text)   | + –            | +10 %          | +17 %          |
| LAMBADA (coherence) | Baseline       | +4 %           | +9 %           |
| Code completion     | Baseline       | +6 %           | +11 %          |
| Throughput          | 1.0 ×          | 0.97 ×         | 0.9 ×          |

---

### 7️⃣ Visual Placeholder

```
[Diagram: Goose State Editing]
Memory S_{t-1} → (decay w_t) → (erase via κ̂_t,a_t) → (add via v_t,κ̃_t) → S_t
```

---

### 🧩 RWKV-7 vs Transformer (Conceptual Bridge)

| Mechanism          | Transformer                         | RWKV-7 (Goose)                         |
| ------------------ | ----------------------------------- | -------------------------------------- |
| *Attention Matrix* | dense QKᵀ softmax                   | implicit through matrix state Sₜ       |
| *Context Update*   | recomputed each token               | recurrent state update                 |
| *Forgetting*       | none (softmax renormalization only) | explicit via (w_t,\hat{\kappa}_t)      |
| *Memory Write*     | new attention weights               | additive (v_t^\top \tilde{\kappa}_t)   |
| *Training*         | fully parallel                      | prefill parallel + recurrent streaming |
| *Inference Memory* | O (T d)                             | O (d² / #heads)                        |

---

### 8️⃣ RWKV-7 vs RNN/LSTM

| Feature           | LSTM                  | Goose                  |
| ----------------- | --------------------- | ---------------------- |
| Hidden state      | vector                | matrix per head        |
| Gates             | input, forget, output | (a_t,w_t,g_t)          |
| Nonlinearity      | tanh/σ                | LoRA-MLP + LayerNorm   |
| Parallel training | ❌                     | ✅                      |
| Expressivity      | low                   | high (state editing)   |
| Gradient flow     | limited               | stable (learned rates) |

---

### 🔚 Key Takeaways — RWKV-6 & RWKV-7

| Property         | Finch                        | Goose                             |
| ---------------- | ---------------------------- | --------------------------------- |
| *Token mixing*   | Data-dependent ddlerp (LoRA) | ddlerp + generalized delta rule   |
| *Decay*          | Dynamic scalar wₜ            | Vector wₜ                         |
| *Memory update*  | Additive decay               | Erase + Add (delta rule)          |
| *Adaptivity*     | Content dependent            | Content & state dependent         |
| *Expressiveness* | High                         | Very High (theoretical NC¹ proof) |
| *Inference cost* | ≈ Eagle                      | ≈ 1.1 × Eagle                     |

Finch and Goose complete the RWKV journey from a “linear attention model” to a full-fledged “neural memory system” capable of dynamic reasoning.

---

**Up next → Part IV:**
🧠 Specialized RWKV variants (Diffusion-RWKV, Vision-RWKV, GoldFinch etc.), global comparisons with Transformers and RNNs, and your “Cheat Sheet Summary.”

Say **“Continue.”** to get the final part.
