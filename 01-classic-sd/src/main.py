import torch

from engine import SpeculativeEngine


def main():
    # Make the first experiment reproducible.
    torch.manual_seed(0)

    decoder = SpeculativeEngine()

    prompt = "Artificial Intelligence is going to"

    # Small smoke test for the memory-limited CPU environment.
    speed, text = decoder.generate(
        prompt,
        max_new_tokens=6,
        gamma=2,
    )

    print("\n--- Final Output ---")
    print(text)

    print("\n--- Performance ---")
    print(f"Generation speed: {speed:.4f} tokens/second")

    # This metric is provisional: we have already identified that
    # the repository may count a replacement token as accepted.
    print("\n--- Repository Metrics (provisional) ---")
    print(decoder.get_last_run_metrics())


if __name__ == "__main__":
    main()