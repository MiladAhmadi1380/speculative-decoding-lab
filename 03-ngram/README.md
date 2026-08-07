# 03 - N-gram Speculative Decoding

This directory studies the native N-gram speculative decoding implementation in vLLM 0.26.0.

## Source

Upstream project:

https://github.com/vllm-project/vllm

Version studied:

vLLM 0.26.0

Relevant source files:

- `src/vllm/ngram_proposer.py`
- `src/vllm/ngram_proposer_gpu.py`

## Main Idea

Unlike draft-model speculative decoding, N-gram speculative decoding does not require a separate draft model.

The proposer searches previously observed token sequences and uses matching continuations as speculative token proposals.

The proposed tokens are then verified by the target model using vLLM's speculative decoding pipeline.

## Previous CPU Experiment

Configuration:

- Model: `facebook/opt-125m`
- Speculative method: `ngram`
- Speculative tokens: 4
- Device: CPU

Recorded result:

```text
speculative rounds = 6
draft tokens       = 24
accepted tokens    = 24
acceptance rate    = 100%
total request time = 105.858728 s