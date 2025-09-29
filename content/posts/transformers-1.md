---
title: "Demystifying Transformers: Attention, Multi-Head Magic, and the Math Behind the Revolution"
date: 2025-09-27
tags: ["Transformers", "Attention", "NLP"]
math: true
---

# Demystifying Transformers: Attention, Multi-Head Magic, and the Math Behind the Revolution

*From single head to multi-head attention - understanding the architectural breakthrough that changed AI forever*

The Transformer architecture, introduced in the seminal "Attention Is All You Need" paper, revolutionized natural language processing by replacing recurrent networks with a purely attention-based approach. At its heart lies the self-attention mechanism - a powerful way for models to understand relationships between all words in a sequence simultaneously.

## The Core Idea: Self-Attention

Self-attention allows each position in a sequence to 'attend' to all other positions, computing a weighted sum of values where the weights are determined by 'compatibility' between queries and keys.

### The Mathematical Foundation

Let's break down the self-attention mechanism with concrete numbers:

**Input**: A sequence of 3 words, each represented as 4-dimensional vectors:

```python
x = [
    [0.2, -0.1, 0.8, 0.4], # Word 1
    [0.5, 0.3, -0.2, 0.1], # Word 2
    [-0.3, 0.7, 0.1, -0.5] # Word 3
    ] 
```

**Step 1: Create Query, Key, Value Matrices**

We learn three weight matrices to transform our input:
- `$W_q$` (Query weights): (4, 3) - transforms input to query space
- `$W_k$` (Key weights): (4, 3) - transforms input to key space  
- `$W_v$` (Value weights): (4, 3) - transforms input to value space

Let's use example weights:

```python
W_q = [[0.1, 0.2, 0.3],
       [0.4, 0.5, 0.6],
       [0.7, 0.8, 0.9],
       [1.0, 1.1, 1.2]]

W_k = [[0.2, 0.3, 0.4],
       [0.5, 0.6, 0.7], 
       [0.8, 0.9, 1.0],
       [1.1, 1.2, 1.3]]

W_v = [[0.3, 0.4, 0.5],
       [0.6, 0.7, 0.8],
       [0.9, 1.0, 1.1],
       [1.2, 1.3, 1.4]]
```

Now compute Q, K, V:

```python
Q = x @ W_q = [[1.3, 1.5, 1.7],   # Queries for each word
               [0.8, 0.9, 1.0],
               [0.7, 0.8, 0.9]]

K = x @ W_k = [[1.5, 1.7, 1.9],   # Keys for each word
               [0.9, 1.0, 1.1],
               [0.8, 0.9, 1.0]]

V = x @ W_v = [[1.7, 1.9, 2.1],   # Values for each word
               [1.0, 1.1, 1.2],
               [0.9, 1.0, 1.1]]
```

**Step 2: Compute Attention Scores**

We calculate how much each word should attend to every other word:

```python
scores = Q @ K.T = [[1.3*1.5 + 1.5*1.7 + 1.7*1.9, ...],
                    ...]
                    
# Result:
scores = [[7.73, 4.53, 4.13],
          [4.53, 2.66, 2.42], 
          [4.13, 2.42, 2.20]]
```


**Step 3: Scale and Softmax**

Scale by $\sqrt{d_k} (\sqrt{3} \approx 1.732)$ and apply softmax:

```python
scaled_scores = scores / 1.732 = [[4.46, 2.62, 2.38],
                                 [2.62, 1.54, 1.40],
                                 [2.38, 1.40, 1.27]]

attention_weights = softmax(scaled_scores, axis=1)
    = [
       [0.70, 0.18, 0.12],
       [0.52, 0.28, 0.20], 
       [0.50, 0.29, 0.21]
      ]
```

**Step 4: Weighted Sum of Values**

Finally, compute the output by weighting values with attention weights:

```python
output = attention_weights @ V
= [[0.70*1.7 + 0.18*1.0 + 0.12*0.9, ...],
   ...]
   
= [[1.43, 1.59, 1.75],
   [1.30, 1.44, 1.58],
   [1.28, 1.42, 1.56]]
```
This output becomes the new representation where each word now contains information about all other relevant words in the sequence!


## The Limitation: Single-Head Attention
Single-head attention has a fundamental limitation - it can only learn one type of relationship pattern. Think of it like having only one perspective when analyzing a sentence.

For example, in the sentence "The bank of the river had money in it", a single attention head might struggle to capture both:

- Syntactic relationships: "bank" is connected to "river" (geographical feature)

- Semantic relationships: "bank" is connected to "money" (financial institution)

## Multi-Head Attention: Multiple Perspectives
Multi-head attention solves this by running multiple attention mechanisms in parallel, each learning different types of relationships.

### Why We Need Multiple Heads
Each attention head can specialize in different aspects. For example:

- Head 1: Focus on syntactic relationships (subject-verb, adjective-noun)

- Head 2: Focus on semantic relationships (synonyms, related concepts)

- Head 3: Focus on long-range dependencies

- Head 4: Focus on positional patterns

This is analogous to how humans analyze text from multiple angles simultaneously.

## Implementation: Two Approaches
There are two common ways to implement multi-head attention:

### Approach 1: Split Large Matrices (Most Common)

We create larger Q, K, V matrices and split them into heads:

```python
# For 2 heads with hidden_dim=4, each head gets 2 dimensions
W_q = (4, 8)  # Instead of (4, 3) for single head
Q = x @ W_q = (3, 8)  # [word1, word2, word3] × 8 dimensions

# Split into 2 heads, each with 4 dimensions
Q_heads = split(Q, 2)  # Two (3, 4) matrices
K_heads = split(K, 2)  # Two (3, 4) matrices  
V_heads = split(V, 2)  # Two (3, 4) matrices

# Compute attention for each head
head1 = attention(Q_heads[0], K_heads[0], V_heads[0])  # (3, 4)
head2 = attention(Q_heads[1], K_heads[1], V_heads[1])  # (3, 4)

# Concatenate and project
multi_head_output = concat([head1, head2]) @ W_o  # (3, 8) → (3, 4)
```

### Approach 2: Separate Matrices for Each Head

We can also use completely separate weight matrices for each head:

```python
# Separate weights for each head
W_q1, W_q2 = (4, 3), (4, 3)  # Two separate query matrices
W_k1, W_k2 = (4, 3), (4, 3)  # Two separate key matrices
W_v1, W_v2 = (4, 3), (4, 3)  # Two separate value matrices

# Compute queries, keys, values for each head
Q1 = x @ W_q1  # (3, 3)
Q2 = x @ W_q2  # (3, 3)
K1 = x @ W_k1  # (3, 3)  
K2 = x @ W_k2  # (3, 3)
V1 = x @ W_v1  # (3, 3)
V2 = x @ W_v2  # (3, 3)

# Compute attention for each head
head1 = attention(Q1, K1, V1)  # (3, 3)
head2 = attention(Q2, K2, V2)  # (3, 3)

# Concatenate and project back to original dimension
concat_heads = concat([head1, head2])  # (3, 6)
W_o = (6, 4)  # Projection matrix
output = concat_heads @ W_o  # (3, 4)
```

**Which approach is better?** Approach 1 (splitting) is more parameter-efficient and is used in most implementations. Approach 2 (separate matrices) gives each head more independence but uses more parameters.

## The Complete Multi-Head Attention Formula

```python
def multi_head_attention(x, num_heads=2):
    # Project to higher dimension for splitting
    Q = x @ W_q  # (seq_len, hidden_dim * num_heads)  
    K = x @ W_k  # (seq_len, hidden_dim * num_heads)
    V = x @ W_v  # (seq_len, hidden_dim * num_heads)
    
    # Split into multiple heads
    Q_heads = split(Q, num_heads)  # list of (seq_len, hidden_dim)
    K_heads = split(K, num_heads)  
    V_heads = split(V, num_heads)
    
    # Compute attention for each head
    heads = []
    for i in range(num_heads):
        head = attention(Q_heads[i], K_heads[i], V_heads[i])
        heads.append(head)
    
    # Concatenate all heads
    concat_heads = concatenate(heads)  # (seq_len, hidden_dim * num_heads)
    
    # Project back to original dimension
    output = concat_heads @ W_o  # (seq_len, hidden_dim)
    return output
```

## Why This Architecture Works So Well

1. **Parallelization**: Unlike RNNs, all attention calculations can happen simultaneously

2. **Global Context**: Each word can directly attend to every other word

3. **Specialization**: Different heads learn different relationship types

4. **Interpretability**: We can analyze what each attention head is learning


## The Trade-off: Computational Cost

The power of multi-head attention comes at a cost - the self-attention mechanism has $O(n^2)$ complexity where n is sequence length. This is why handling very long sequences remains challenging, motivating research into efficient attention variants.

The multi-head attention mechanism demonstrates the power of learning multiple specialized perspectives - a principle that extends beyond transformers to how we might approach complex problems in general.