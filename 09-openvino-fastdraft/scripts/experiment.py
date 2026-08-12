import csv
import gc
from pathlib import Path
from statistics import mean

import openvino_genai as ov_genai


TARGET_MODEL = (
    r"C:\Users\milad\sd-lab\models\openvino"
    r"\Qwen2-1.5B-Instruct-int4-ov"
)

DRAFT_MODEL = (
    r"C:\Users\milad\sd-lab\models\openvino"
    r"\Qwen2-0.5B-Instruct-int4-ov"
)

DEVICE = "CPU"

PROMPT = (
    "Explain speculative decoding in five numbered steps. "
    "Include drafting, target verification, token acceptance, "
    "token rejection, and why speculation length matters."
)

MAX_NEW_TOKENS = 64

NUM_WARMUP = 1
NUM_REPEATS = 3

K_VALUES = [2, 3, 5, 8]

RESULTS_FILE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "k_sweep.csv"
)


def make_scheduler():
    scheduler = ov_genai.SchedulerConfig()

    scheduler.enable_prefix_caching = False
    scheduler.cache_size = 1

    return scheduler


def make_generation_config(k=None):
    config = ov_genai.GenerationConfig()

    config.max_new_tokens = MAX_NEW_TOKENS
    config.do_sample = False

    if k is not None:
        config.num_assistant_tokens = k

    return config


def build_target_pipeline():
    print("\nBuilding target-only pipeline...")

    return ov_genai.LLMPipeline(
        TARGET_MODEL,
        DEVICE,
        scheduler_config=make_scheduler(),
    )


def build_speculative_pipeline():
    print("\nLoading draft model...")

    draft = ov_genai.draft_model(
        DRAFT_MODEL,
        DEVICE,
    )

    print("Building speculative pipeline...")

    return ov_genai.LLMPipeline(
        TARGET_MODEL,
        DEVICE,
        scheduler_config=make_scheduler(),
        draft_model=draft,
    )


def run_generation(pipe, config, mode, k, repeat):
    result = pipe.generate(
        [PROMPT],
        config,
    )

    metrics = result.perf_metrics

    row = {
        "mode": mode,
        "k": "" if k is None else k,
        "repeat": repeat,
        "generated_tokens": metrics.get_num_generated_tokens(),
        "generate_ms": metrics.get_generate_duration().mean,
        "ttft_ms": metrics.get_ttft().mean,
        "tpot_ms": metrics.get_tpot().mean,
        "throughput_tps": metrics.get_throughput().mean,
        "output": result.texts[0],
    }

    print(
        f"{mode:12s} "
        f"K={str(k):>4s} "
        f"run={repeat} | "
        f"tokens={row['generated_tokens']:3d} | "
        f"time={row['generate_ms'] / 1000:.2f}s | "
        f"TPOT={row['tpot_ms']:.2f}ms | "
        f"throughput={row['throughput_tps']:.2f} tok/s"
    )

    return row


def warm_up(pipe, config, label):
    print(f"Warming up {label}...")

    for _ in range(NUM_WARMUP):
        pipe.generate(
            [PROMPT],
            config,
        )


def write_results(rows):
    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "mode",
        "k",
        "repeat",
        "generated_tokens",
        "generate_ms",
        "ttft_ms",
        "tpot_ms",
        "throughput_tps",
        "output",
    ]

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print("\n================ SUMMARY ================")

    baseline_rows = [
        row
        for row in rows
        if row["mode"] == "baseline"
    ]

    baseline_time = mean(
        row["generate_ms"]
        for row in baseline_rows
    )

    baseline_throughput = mean(
        row["throughput_tps"]
        for row in baseline_rows
    )

    print(
        f"Baseline: "
        f"{baseline_time / 1000:.2f}s | "
        f"{baseline_throughput:.2f} tok/s"
    )

    for k in K_VALUES:
        k_rows = [
            row
            for row in rows
            if row["mode"] == "static_sd"
            and row["k"] == k
        ]

        k_time = mean(
            row["generate_ms"]
            for row in k_rows
        )

        k_throughput = mean(
            row["throughput_tps"]
            for row in k_rows
        )

        speedup = baseline_time / k_time

        print(
            f"K={k}: "
            f"{k_time / 1000:.2f}s | "
            f"{k_throughput:.2f} tok/s | "
            f"speedup={speedup:.3f}x"
        )


def check_outputs(rows):
    baseline_output = next(
        row["output"]
        for row in rows
        if row["mode"] == "baseline"
    )

    mismatches = []

    for row in rows:
        if row["output"] != baseline_output:
            mismatches.append(
                (
                    row["mode"],
                    row["k"],
                    row["repeat"],
                )
            )

    print("\nOutput equivalence check:")

    if not mismatches:
        print("PASS - all measured outputs match baseline.")
    else:
        print("WARNING - output mismatch detected:")

        for mismatch in mismatches:
            print(mismatch)


def main():
    print("OpenVINO GenAI - Static Speculation K Sweep")
    print(f"Device: {DEVICE}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Measured repetitions: {NUM_REPEATS}")
    print(f"K values: {K_VALUES}")

    rows = []

    # -----------------------------------------
    # Target-only baseline
    # -----------------------------------------

    target_pipe = build_target_pipeline()

    baseline_config = make_generation_config()

    warm_up(
        target_pipe,
        baseline_config,
        "target-only baseline",
    )

    for repeat in range(
        1,
        NUM_REPEATS + 1,
    ):
        rows.append(
            run_generation(
                target_pipe,
                baseline_config,
                "baseline",
                None,
                repeat,
            )
        )

    del target_pipe
    gc.collect()

    # -----------------------------------------
    # Static speculative decoding
    # -----------------------------------------

    speculative_pipe = build_speculative_pipeline()

    for k in K_VALUES:
        config = make_generation_config(k=k)

        warm_up(
            speculative_pipe,
            config,
            f"static SD K={k}",
        )

        for repeat in range(
            1,
            NUM_REPEATS + 1,
        ):
            rows.append(
                run_generation(
                    speculative_pipe,
                    config,
                    "static_sd",
                    k,
                    repeat,
                )
            )

    write_results(rows)

    check_outputs(rows)

    print_summary(rows)

    print(
        f"\nCSV written to:\n{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()