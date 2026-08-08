# Lookahead Decoding

Official source implementation of Lookahead Decoding.

Upstream:
https://github.com/hao-ai-lab/LookaheadDecoding

## Core idea

Lookahead Decoding removes the need for a separate draft model.

It uses Jacobi-style parallel decoding to generate candidate n-grams, stores useful candidates in a token map, and verifies them with the language model.

Simplified flow:

Jacobi lookahead windows
-> candidate n-grams
-> token_map
-> verification
-> longest accepted sequence
-> commit

## Key source file

`src/LookaheadDecoding/lade/decoding.py`

Important functions:

- `update_token_map()`
- `append_new_generated_pool()`
- `fill_pool_with_prompt()`
- `jacobi_sample_multilevel()`
- `jacobi_greedy_search_multilevel()`

During greedy verification:

- Candidate n-grams are retrieved from `token_map`.
- `outputs.guess_logits` are used for verification.
- `max_hit` represents the length of the longest accepted speculative sequence.
- Accepted KV-cache entries and tokens are committed to the sequence.

## Status

- Official source collected
- Jacobi candidate generation inspected
- N-gram pool inspected
- Greedy and sampling verification logic inspected

The original upstream LICENSE and attribution are preserved inside `src/LookaheadDecoding/`.