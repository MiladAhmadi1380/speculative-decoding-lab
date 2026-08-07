import torch
import torch.nn.functional as F


def sample_token(
    logits,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
):
    """Sample one token from a processed logits distribution."""
    logits = logits / (temperature + 1e-10)

    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k)
        min_value = top_k_values[..., -1].unsqueeze(-1)
        negative_infinity = torch.tensor(
            float("-inf"),
            device=logits.device,
            dtype=logits.dtype,
        )
        logits = torch.where(
            logits < min_value,
            negative_infinity,
            logits,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True,
        )
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1),
            dim=-1,
        )

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[
            sorted_indices_to_remove
        ]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)

    if torch.isnan(probs).any():
        probs = torch.ones_like(probs) / probs.shape[-1]

    next_token = torch.multinomial(
        probs,
        num_samples=1,
    )
    return next_token.item()


def verify_and_sample(
    target_logits,
    draft_logits,
    draft_token,
    temperature=1.0,
):
    """
    Verify one draft token.

    Returns:
        accepted:
            Whether the proposed draft token was accepted.

        final_token:
            The accepted draft token, or a replacement sampled
            from the residual target distribution.

        trace:
            Numeric details of the acceptance decision.
    """
    target_probs = F.softmax(
        target_logits / (temperature + 1e-10),
        dim=-1,
    )
    draft_probs = F.softmax(
        draft_logits / (temperature + 1e-10),
        dim=-1,
    )

    p_target = target_probs[draft_token].item()
    p_draft = draft_probs[draft_token].item()

    acceptance_prob = min(
        1.0,
        p_target / (p_draft + 1e-10),
    )

    random_draw = torch.rand(
        (),
        device=target_logits.device,
    ).item()

    trace = {
        "p_target": p_target,
        "p_draft": p_draft,
        "acceptance_probability": acceptance_prob,
        "random_draw": random_draw,
        "residual_mass": None,
        "replacement_token": None,
    }

    if random_draw < acceptance_prob:
        return True, draft_token, trace

    residual_probs = torch.clamp(
        target_probs - draft_probs,
        min=0.0,
    )
    residual_mass = residual_probs.sum()

    trace["residual_mass"] = residual_mass.item()

    # This fallback only protects against a degenerate
    # all-zero residual distribution.
    if residual_mass.item() <= 1e-12:
        residual_probs = target_probs
    else:
        residual_probs = residual_probs / residual_mass

    new_token = torch.multinomial(
        residual_probs,
        num_samples=1,
    ).item()

    trace["replacement_token"] = new_token

    return False, new_token, trace