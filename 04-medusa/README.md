# 04 - Medusa Speculative Decoding

This directory contains the Medusa-related source code from vLLM 0.26.0.

## Upstream

https://github.com/vllm-project/vllm

Version studied:

vLLM 0.26.0

## Source Files

- `src/vllm/medusa.py`
  - Medusa proposer used by the vLLM speculative decoding pipeline.

- `src/vllm/medusa_model.py`
  - Implementation of Medusa residual blocks and LM heads.

- `src/vllm/medusa_config.py`
  - Medusa model configuration.

## Main Idea

Medusa does not use a separate autoregressive draft language model.

Instead, hidden states produced by the target model are passed to multiple
Medusa heads.

Each head predicts a token for a future position.

Simplified pipeline:

Target hidden states
→ Medusa heads
→ logits for future positions
→ top-1 token from each head
→ speculative verification

## vLLM 0.26.0 Implementation

The proposer computes:

```python
blocks = self.model(target_hidden_states)
logits = self.model.compute_logits(blocks)

draft_tokens = torch.stack(
    [logit.argmax(dim=-1) for logit in logits],
    dim=1
)

The studied implementation currently uses the top-1 token produced by each
Medusa head for speculation.

Key Configuration

Typical Medusa configuration fields include:

num_heads
num_hidden_layers
max_paths
topk

The default configuration in the studied source contains five Medusa heads.