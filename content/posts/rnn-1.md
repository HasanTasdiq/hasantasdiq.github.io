---
title: "Demystifying RNNs: A Deep Dive into Dimensions and Parameters"
date: 2025-09-26
tags: ["RNN", "Neural Network", "Machine Learning", "NLP"]
math: true
---


# Demystifying RNNs: A Deep Dive into Dimensions and Parameters

*Understanding what really happens inside Recurrent Neural Networks*

When learning about Recurrent Neural Networks (RNNs), many tutorials focus on the high-level concept of "memory" but gloss over the practical details of how they actually work. As someone who struggled with these details, I want to share the insights that finally made RNNs click for me.

## The Core RNN Equations

Let's start with the fundamental RNN equations that everyone shows:

$$
h_t = tanh(W_{hh} · h_{t-1} + W_{hx} \cdot x_t + b_h)
\\
y_t = W_{ho} \cdot h_t + b_v
$$


These equations look simple enough, but the devil is in the dimensions. Let's break them down with a concrete example.

## A Concrete Example

Let's define our dimensions:
- **Input dimension** (`$d_{in}$`): 4 (each input is a 4D vector)
- **Hidden dimension** (`$d_h$`): 3 (the size of our RNN's "memory")
- **Output dimension** (`$d_{out}$`): 2 (e.g., binary classification)

Now let's look at what each component actually contains:

### The Vectors (Changing States)

| Component | Shape | Description |
|-----------|-------|-------------|
| `$x_t$` | `(4,)` | Input at time t (e.g., a word embedding) |
| `$h_{t-1}$` | `(3,)` | Previous hidden state (the "memory" so far) |
| `$h_t$` | `(3,)` | New hidden state (updated memory) |
| `$y_t$` | `(2,)` | Output at time t |


**Key Insight**: `$h_t$` and `$x_t$` do NOT have the same dimension! The hidden dimension is a design choice, while input dimension is determined by your data.

### The Parameters (Learned Weights)

| Component | Shape | Purpose |
|-----------|-------|---------|
| `$W_{hx}$` | `(3, 4)` | Transforms input to hidden space |
| `$W_{hh}$` | `(3, 3)` | Transforms previous hidden state |
| `$b_h$` | `(3,)` | Hidden layer bias |
| `$W_{ho}$` | `(2, 3)` | Transforms hidden state to output |
| `$b_v$` | `(2,)` | Output bias |

**Key Insight**: The weight matrices are the "bridges" that make different dimensions compatible. They're the actual parameters learned during training.

## Dimensionality Check: Why It All Works

Let's verify the math works dimensionally:

```python
# All operations are dimensionally compatible:
W_hh DOT h_{t-1}    # (3,3) DOT (3,)  → (3,)
W_hx DOT x_t         # (3,4) DOT (4,)  → (3,) 
b_h                # (3,)
# Sum: (3,) + (3,) + (3,) → (3,)
tanh(...)          # (3,) → (3,)  # h_t is born!

W_ho DOT h_t         # (2,3) DOT (3,) → (2,)
b_v                # (2,)
# Sum: (2,) + (2,) → (2,)  # y_t is ready!
```

## Common Questions Answered
### 1. Is $h_t$ a vector or matrix?
In the fundamental formulation, $h_t$ is a vector. However, during batch processing (which we almost always do), it becomes a matrix where each row is the $h_t$ for one sequence in the batch.

### 2. How is $h_0$ initialized?
Typically with zeros: $h_0 = [0, 0, 0, ..., 0]$. This provides a neutral starting point.

### 3. What's actually being "learned"?
The weight matrices and biases $(W_{hh}, W_{hx}, W_{ho}, b_h, b_v)$ are the learned parameters. The hidden state $h_t$ is the result of computation, not a parameter.

### 4. Why can RNNs handle variable-length sequences?
Because the same parameters (weights) are reused at each time step, and the hidden state dimension remains constant regardless of sequence length.

## Parameter Counting
In our example:

$W_{hx}: 3 \times 4 = 12$ parameters

$W_{hh}: 3 \times 3 = 9$ parameters

$b_h: 3$ parameters

$W_{ho}: 2 \times 3 = 6$ parameters

$b_v: 2$ parameters

Total: 32 parameters (regardless of sequence length!)

## The Achilles' Heel: Vanishing and Exploding Gradients

Despite their elegant design, RNNs suffer from a fundamental limitation: they struggle to learn long-term dependencies. This occurs due to the vanishing and exploding gradient problem.

During training, gradients are calculated and propagated backward through time. At each step, the gradient gets multiplied by the same weight matrix $W_{hh}$. The behavior of this repeated multiplication depends on the eigenvalues of $W_{hh}$.

**What are eigenvalues?** Think of them as the matrix's "scaling factors" -- they tell you how much a vector gets stretched or compressed when multiplied by the matrix.

- If the largest eigenvalue is less than 1: Gradients shrink exponentially as they backpropagate through time, eventually vanishing to near-zero. The network loses its ability to learn from distant time steps.

- If the largest eigenvalue is greater than 1: Gradients grow exponentially, exploding to enormous values and making training unstable.

This fragility stems from RNNs' sequential structure: each hidden state depends solely on its immediate predecessor. The result is a brittle information chain where long-range dependencies vanish, limiting the model's ability to capture context across extended sequences.


## The Big Picture
Think of an RNN as a function: $h_t = f(x_t, h_{t-1})$

Parameters = The fixed "knobs" of the function (weights and biases)

Hidden state = The changing "memory" that gets passed between function calls

The magic = The same function f is called repeatedly, each time updating the memory based on new input

## Why This Matters
Understanding these dimensional relationships is crucial because:

1. It helps debug shape errors when implementing RNNs
2. It clarifies what the model is actually learning
3. It provides intuition for more advanced architectures (LSTMs, GRUs, Transformers)
4. It explains why RNNs can handle sequences of any length

The next time you see RNN equations, remember: the dimensions tell the real story! The matrices are the bridges that make everything connect, and the hidden state is the messenger carrying information through time.
