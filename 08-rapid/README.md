# 08 - RAPID: Retrieval-Augmented Speculative Decoding

Implementation study and experimental extension of RAPID:

> RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding

Official upstream:
https://github.com/NUS-TRAIL/RAPID

Upstream commit inspected:
`22d41f4`

## Goal

Understand and extend RAPID for experiments on joint adaptation of:

- `B`: RAG drafter context/evidence budget
- `K`: speculative draft length (`num_assistant_tokens`)

The long-context target model retains the full context, while the
RAG drafter operates on a shorter retrieved context.

## RAPID code path

### Draft-context path

Long context
→ chunking
→ embedding retrieval
→ retrieved chunks
→ RAG prompt
→ assistant input IDs
→ RAG drafter prefill

### Speculative-length path

`num_assistant_tokens`
→ generation config
→ RAPIDAssistedCandidateGenerator
→ `self.num_assistant_tokens`
→ candidate generation
→ target verification

## Planned extension

Expose the experimental configuration explicitly:

- draft context budget B
- speculative length K

Example target configuration:

B = 4096 tokens
K = 8 tokens

Future experiments will sweep B × K and measure:

- candidate tokens
- accepted tokens
- acceptance rate
- prefill latency
- decoding latency
- tokens/s

## Status

- [x] Official RAPID repository inspected
- [x] Retrieval/context path identified
- [x] Speculative-length path identified
- [x] Existing acceptance metrics identified
- [ ] Add explicit draft-context-budget control
- [ ] Add experiment configuration logging
- [ ] Functional CPU test
- [ ] GPU B × K benchmark