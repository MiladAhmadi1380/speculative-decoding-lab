import time
import openvino_genai as ov_genai


TARGET_MODEL = r"C:\Users\milad\sd-lab\models\openvino\Qwen2-1.5B-Instruct-int4-ov"
DRAFT_MODEL = r"C:\Users\milad\sd-lab\models\openvino\Qwen2-0.5B-Instruct-int4-ov"

DEVICE = "CPU"
PROMPT = "Explain speculative decoding in one sentence."
MAX_NEW_TOKENS = 32


def make_scheduler():
    scheduler = ov_genai.SchedulerConfig()

    # Keep the benchmark simple and avoid reusing prefixes between runs.
    scheduler.enable_prefix_caching = False

    # Small cache is enough for our single-request CPU experiment.
    scheduler.cache_size = 1

    return scheduler


def build_target_pipeline():
    print("Building target-only pipeline...")

    start = time.perf_counter()

    pipe = ov_genai.LLMPipeline(
        TARGET_MODEL,
        DEVICE,
        scheduler_config=make_scheduler(),
    )

    load_seconds = time.perf_counter() - start

    print(f"Target pipeline ready in {load_seconds:.2f} s")
    return pipe


def build_speculative_pipeline():
    print("Loading draft model...")

    draft = ov_genai.draft_model(
        DRAFT_MODEL,
        DEVICE,
    )

    print("Building speculative pipeline...")

    start = time.perf_counter()

    pipe = ov_genai.LLMPipeline(
        TARGET_MODEL,
        DEVICE,
        scheduler_config=make_scheduler(),
        draft_model=draft,
    )

    load_seconds = time.perf_counter() - start

    print(f"Speculative pipeline ready in {load_seconds:.2f} s")
    return pipe


def make_generation_config(k=None):
    config = ov_genai.GenerationConfig()

    config.max_new_tokens = MAX_NEW_TOKENS

    # We keep decoding greedy for this experiment.
    config.do_sample = False

    if k is not None:
        config.num_assistant_tokens = k

    return config


def run_once(pipe, label, config):
    # Passing a list makes OpenVINO return DecodedResults,
    # which exposes perf_metrics.
    result = pipe.generate([PROMPT], config)

    metrics = result.perf_metrics

    print()
    print(f"=== {label} ===")
    print(f"Output: {result.texts[0]}")
    print(f"Generated tokens: {metrics.get_num_generated_tokens()}")
    print(f"Generate duration: {metrics.get_generate_duration().mean:.2f} ms")
    print(f"TTFT: {metrics.get_ttft().mean:.2f} ms")
    print(f"TPOT: {metrics.get_tpot().mean:.2f} ms/token")
    print(f"Throughput: {metrics.get_throughput().mean:.2f} tokens/s")

    return metrics


def main():
    print("OpenVINO GenAI speculative decoding mini-project")
    print(f"Target: {TARGET_MODEL}")
    print(f"Draft:  {DRAFT_MODEL}")
    print(f"Device: {DEVICE}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print()

    target_pipe = build_target_pipeline()
    baseline_config = make_generation_config()

    print("\nWarming up target-only pipeline...")
    target_pipe.generate([PROMPT], baseline_config)

    run_once(
        target_pipe,
        "TARGET ONLY",
        baseline_config,
    )

    # Release the target-only pipeline before constructing another
    # target + draft pipeline on an 8 GB machine.
    del target_pipe

    speculative_pipe = build_speculative_pipeline()
    speculative_config = make_generation_config(k=3)

    print("\nWarming up speculative pipeline...")
    speculative_pipe.generate([PROMPT], speculative_config)

    run_once(
        speculative_pipe,
        "STATIC SPECULATIVE DECODING (K=3)",
        speculative_config,
    )


if __name__ == "__main__":
    main()