# LayerSkip Self-Speculative Decoding

Official source implementation of LayerSkip from Meta.

Upstream:
https://github.com/facebookresearch/LayerSkip

## Core idea

LayerSkip performs self-speculative decoding using a single language model.

- Early layers generate draft tokens.
- The remaining layers verify the drafts.
- The longest accepted prefix is committed.
- No separate draft language model is required.

## Key source files

- `src/LayerSkip/self_speculation/self_speculation_generator.py`
  - Main self-speculative decoding loop
  - Draft generation
  - Verification
  - Acceptance and commit

- `src/LayerSkip/self_speculation/llama_model_utils.py`
  - `forward_early()`
  - `forward_remainder()`
  - `decode_next_token()`

## Status

- Official source collected
- Core implementation inspected
- Execution postponed because of local hardware/environment constraints

The original upstream LICENSE and attribution are preserved inside `src/LayerSkip/`.
