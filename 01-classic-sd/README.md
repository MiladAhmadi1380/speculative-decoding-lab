# 01 - Classic Speculative Decoding

A hands-on implementation of classic speculative decoding used for studying
the draft-target verification pipeline.

## Original Source

Original repository:

https://github.com/kunal51107/Speculative-decoding-engine

The original implementation was modified during this study for CPU execution,
correct metric calculation, token-level tracing, and full-acceptance handling.

## Models

- Target: `facebook/opt-350m`
- Draft: `facebook/opt-125m`
- Device: CPU
- Gamma: 2
- Temperature: 1.0
- Top-k: 0
- Top-p: 1.0

## Main Code

- `src/main.py` — experiment entry point
- `src/engine.py` — speculative decoding pipeline
- `src/sampling.py` — sampling and acceptance/rejection logic
- `src/config.py` — model and sampling configuration

Additional experimental files:

- `src/baseline.py`
- `src/compare.py`
- `src/experiments.py`

## Pipeline

Draft generation  
→ Target verification  
→ Acceptance / rejection sampling  
→ Replacement on rejection  
→ Commit valid tokens  
→ Next speculative round

## Changes Made

The implementation was modified to:

1. Fix incorrect acceptance-rate accounting.
2. Separate accepted, rejected, and replacement token metrics.
3. Add token-level tracing including:
   - draft probability
   - target probability
   - acceptance probability
   - random draw
   - ACCEPT / REJECT result
   - residual mass
   - replacement token
4. Add the bonus target token when all draft proposals in a round are accepted.

## Example Final Run

```text
total_draft_tokens       = 5
accepted_draft_tokens    = 4
rejected_draft_tokens    = 1
replacement_tokens       = 1
bonus_target_tokens      = 1
total_committed_tokens   = 6
acceptance_rate          = 80%