@echo off

docker run --rm --name vllm-cpu-ngram ^
  --security-opt seccomp=unconfined ^
  --cap-add SYS_NICE ^
  --shm-size=1g ^
  -p 8000:8000 ^
  -e VLLM_CPU_KVCACHE_SPACE=1 ^
  -e VLLM_CPU_NUM_OF_RESERVED_CPU=1 ^
  -v "%USERPROFILE%\.cache\huggingface:/root/.cache/huggingface" ^
  vllm/vllm-openai-cpu:latest-x86_64 ^
  facebook/opt-125m ^
  --dtype=bfloat16 ^
  --max-model-len=256 ^
  --max-num-seqs=1 ^
  --enforce-eager ^
  --spec-method ngram ^
  --spec-tokens 4