# 05 - EAGLE / EAGLE-3 Speculative Decoding

This directory contains the main EAGLE and EAGLE-3 speculative decoding
source files from vLLM 0.26.0.

## Upstream

https://github.com/vllm-project/vllm

Version studied:

vLLM 0.26.0

## Source Files

- `eagle.py` - EAGLE proposer integration
- `eagle_speculator.py` - EAGLE speculator
- `autoregressive_speculator.py` - multi-step autoregressive draft generation
- `speculator_base.py` - common draft-speculator infrastructure
- `eagle_utils.py` - EAGLE model loading and target-model sharing
- `eagle3_utils.py` - EAGLE-3 auxiliary hidden-state configuration
- `eagle_config.py` - configuration

## Main Idea

EAGLE uses a learned speculative model that receives information from the
target model's hidden states.

Simplified pipeline:

Target model
→ target hidden states
→ EAGLE speculator
→ autoregressive draft tokens
→ target verification

Unlike classic draft-model speculative decoding, the EAGLE speculator is
conditioned on hidden representations produced by the target model.

## EAGLE-3

In the studied vLLM implementation, EAGLE-3 can use auxiliary hidden states
from multiple target-model layers.

These hidden states are combined before being passed to the speculative model.

Simplified difference:

EAGLE:
last target hidden state
→ speculator

EAGLE-3:
multiple target hidden states
→ combine hidden states
→ speculator

## vLLM Implementation

The EAGLE proposer enables passing target hidden states to the speculative
model:

```python
pass_hidden_states_to_model=True

The EAGLE speculator extends vLLM's common autoregressive speculator:

class EagleSpeculator(AutoRegressiveSpeculator):

Multi-step draft generation repeatedly samples a draft token and updates the
draft inputs for the next speculative step.