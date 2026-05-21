# nanoLM

A character-level language model built from scratch in PyTorch,
following the full progression from bigram model to transformer.
Built to understand how modern language models work at every level
of the stack — from raw character counts to deep MLP architectures
with principled initialization and batch normalization.

## Progression

- **Bigram model** — count-based N-gram and its neural network
  equivalent. Introduces tokenization, the N tensor, softmax,
  negative log likelihood loss, and the training loop.

- **MLP language model** — Bengio et al. 2003 architecture.
  Character embeddings, 3-character context window, hidden layers,
  train/val/test splits, and minibatch gradient descent.

- **Deep MLP with initialization + batch normalization** —
  Xavier and Kaiming initialization, saturated/dead neuron
  diagnostics, batch normalization (Ioffe & Szegedy 2015),
  running statistics for inference, and modular layer classes
  mirroring PyTorch's nn.Linear and nn.BatchNorm1d.

- **Manual backpropagation** — full derivative derivation through
  softmax, cross-entropy, batch norm, and linear layers by hand.
  Validates against PyTorch autograd.

- **Transformer** — in progress

## Key concepts covered

Tokenization, embeddings, softmax, cross-entropy loss, minibatch
gradient descent, Xavier/Kaiming initialization, batch normalization,
train/val/test evaluation, modular PyTorch-style layer design,
manual backpropagation via chain rule

## References

- Bengio et al. 2003 — A Neural Probabilistic Language Model
- Glorot & Bengio 2010 — Xavier initialization
- He et al. 2015 — Kaiming initialization
- Ioffe & Szegedy 2015 — Batch Normalization
- Inspired by [Andrej Karpathy's makemore](https://github.com/karpathy/makemore)
