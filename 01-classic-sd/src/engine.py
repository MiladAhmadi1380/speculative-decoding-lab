import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    DEVICE,
    DRAFT_MODEL_NAME,
    TARGET_MODEL_NAME,
    TEMPERATURE,
    TOP_K,
    TOP_P,
)
from sampling import sample_token, verify_and_sample


class SpeculativeEngine:
    def __init__(self):
        print(f"Initializing engine on {DEVICE}...")
        self.device = DEVICE

        self.tokenizer = AutoTokenizer.from_pretrained(
            TARGET_MODEL_NAME
        )

        print(f"Loading target model ({TARGET_MODEL_NAME})...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            TARGET_MODEL_NAME
        ).to(self.device)
        self.target_model.eval()

        print(f"Loading draft model ({DRAFT_MODEL_NAME})...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_NAME
        ).to(self.device)
        self.draft_model.eval()

        self.metrics = self._empty_metrics()

        print("Engine ready.")

    @staticmethod
    def _empty_metrics():
        return {
            "rounds": 0,
            "full_accept_rounds": 0,
            "total_draft_tokens": 0,
            "total_verified_draft_tokens": 0,
            "total_accepted_draft_tokens": 0,
            "total_rejected_draft_tokens": 0,
            "total_replacement_tokens": 0,
            "total_bonus_target_tokens": 0,
            "total_committed_tokens": 0,
            "acceptance_rate": 0.0,
        }

    def get_last_run_metrics(self):
        """Return metrics from the most recent generation."""
        return self.metrics.copy()

    def _generate_draft_tokens(self, input_ids, gamma):
        """
        Generate speculative proposal tokens autoregressively
        using the smaller draft model.
        """
        draft_tokens = []
        draft_logits_list = []
        current_input = input_ids.clone()

        for _ in range(gamma):
            with torch.no_grad():
                outputs = self.draft_model(current_input)
                next_token_logits = outputs.logits[0, -1, :]

            next_token_id = sample_token(
                next_token_logits,
                TEMPERATURE,
                TOP_K,
                TOP_P,
            )

            draft_tokens.append(next_token_id)
            draft_logits_list.append(next_token_logits)

            next_token_tensor = torch.tensor(
                [[next_token_id]],
                device=self.device,
                dtype=input_ids.dtype,
            )

            current_input = torch.cat(
                [current_input, next_token_tensor],
                dim=1,
            )

        return draft_tokens, draft_logits_list

    def _verify_tokens(
        self,
        input_ids,
        draft_tokens,
        draft_logits_list,
        allow_bonus,
    ):
        """
        Verify all draft proposals with one target-model
        forward pass.

        committed_tokens can contain:
        1. accepted draft tokens;
        2. one replacement token sampled from the target
           residual distribution after the first rejection;
        3. one bonus target token if every draft proposal
           is accepted.
        """
        committed_tokens = []

        accepted_count = 0
        rejected_count = 0
        replacement_count = 0
        verified_count = 0
        bonus_count = 0

        draft_tensor = torch.tensor(
            [draft_tokens],
            device=self.device,
            dtype=input_ids.dtype,
        )

        full_input = torch.cat(
            [input_ids, draft_tensor],
            dim=1,
        )

        # The target verifies the complete draft block
        # in one forward pass.
        with torch.no_grad():
            target_outputs = self.target_model(full_input)
            target_logits_full = target_outputs.logits[0]

        # For a causal LM, logits at the final prompt position
        # predict the first draft token.
        start_pos = input_ids.shape[1] - 1

        for position, draft_token_id in enumerate(draft_tokens):
            current_target_logits = target_logits_full[
                start_pos + position
            ]

            current_draft_logits = draft_logits_list[position]

            accepted, final_token, trace = verify_and_sample(
                current_target_logits,
                current_draft_logits,
                draft_token_id,
                TEMPERATURE,
            )

            verified_count += 1
            committed_tokens.append(final_token)

            draft_text = self.tokenizer.decode(
                [draft_token_id],
                skip_special_tokens=False,
            )

            final_text = self.tokenizer.decode(
                [final_token],
                skip_special_tokens=False,
            )

            decision = "ACCEPT" if accepted else "REJECT"

            print(
                f"  Position {position + 1}: "
                f"draft_id={draft_token_id}, "
                f"draft={draft_text!r}"
            )

            print(
                f"    p_draft={trace['p_draft']:.8f}, "
                f"p_target={trace['p_target']:.8f}"
            )

            print(
                f"    acceptance_probability="
                f"{trace['acceptance_probability']:.8f}, "
                f"random_draw={trace['random_draw']:.8f}"
            )

            print(f"    decision={decision}")

            if accepted:
                accepted_count += 1

            else:
                rejected_count += 1
                replacement_count += 1

                print(
                    f"    replacement_id={final_token}, "
                    f"replacement={final_text!r}, "
                    f"residual_mass="
                    f"{trace['residual_mass']:.8f}"
                )

            # Stop generation if EOS has been committed.
            if final_token == self.tokenizer.eos_token_id:
                break

            # After the first rejection, all later draft
            # proposals become invalid because they were
            # conditioned on the rejected draft token.
            if not accepted:
                break

        # -------------------------------------------------
        # BONUS TARGET TOKEN
        # -------------------------------------------------
        #
        # If every draft proposal was accepted, then the
        # final logits produced by the SAME target forward
        # pass already represent the distribution of the
        # token immediately after the accepted draft block.
        #
        # Therefore we can sample one additional token from
        # the target without performing another target
        # forward pass.
        # -------------------------------------------------

        all_draft_tokens_accepted = (
            verified_count == len(draft_tokens)
            and accepted_count == len(draft_tokens)
            and rejected_count == 0
        )

        eos_was_committed = (
            self.tokenizer.eos_token_id
            in committed_tokens
        )

        if (
            all_draft_tokens_accepted
            and allow_bonus
            and not eos_was_committed
        ):
            # Example:
            #
            # prompt length = L
            # gamma = 2
            #
            # target_logits_full[L - 1]
            #     -> distribution for d1
            #
            # target_logits_full[L]
            #     -> distribution for d2
            #
            # target_logits_full[L + 1]
            #     -> distribution after d2
            #     -> BONUS TOKEN
            #
            # Since:
            #
            # start_pos = L - 1
            #
            # bonus position is:
            #
            # start_pos + len(draft_tokens)

            bonus_logits_index = (
                start_pos + len(draft_tokens)
            )

            bonus_target_logits = target_logits_full[
                bonus_logits_index
            ]

            bonus_token = sample_token(
                bonus_target_logits,
                TEMPERATURE,
                TOP_K,
                TOP_P,
            )

            committed_tokens.append(bonus_token)
            bonus_count = 1

            bonus_text = self.tokenizer.decode(
                [bonus_token],
                skip_special_tokens=False,
            )

            print(
                f"  Bonus target token: "
                f"id={bonus_token}, "
                f"token={bonus_text!r}"
            )

        return {
            "committed_tokens": committed_tokens,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "replacement_count": replacement_count,
            "verified_count": verified_count,
            "bonus_count": bonus_count,
        }

    def generate(
        self,
        prompt,
        max_new_tokens=30,
        gamma=4,
    ):
        """
        Generate text using classic speculative decoding.

        Args:
            prompt:
                Input text.

            max_new_tokens:
                Maximum number of output tokens.

            gamma:
                Maximum number of draft tokens proposed
                in each speculative round.

        Returns:
            speed:
                Generated tokens per second.

            text:
                Decoded prompt and generated output.
        """
        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive."
            )

        if gamma <= 0:
            raise ValueError(
                "gamma must be positive."
            )

        run_metrics = self._empty_metrics()

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device).input_ids

        original_input_len = input_ids.shape[1]

        num_generated = 0
        round_number = 0

        start_time = time.perf_counter()

        while num_generated < max_new_tokens:
            round_number += 1

            # Number of output tokens still allowed.
            remaining_tokens = (
                max_new_tokens - num_generated
            )

            # Do not propose more draft tokens than the
            # remaining generation budget.
            current_gamma = min(
                gamma,
                remaining_tokens,
            )

            # A bonus token is allowed only if there is still
            # room after committing all current draft tokens.
            #
            # Example:
            #
            # remaining = 6
            # gamma = 2
            # -> bonus allowed
            #
            # remaining = 2
            # gamma = 2
            # -> bonus NOT allowed
            allow_bonus = (
                remaining_tokens > current_gamma
            )

            print(
                f"\n=== Speculative Round {round_number} ==="
            )

            # ---------------------------------------------
            # STEP 1: DRAFT PROPOSAL
            # ---------------------------------------------

            draft_tokens, draft_logits = (
                self._generate_draft_tokens(
                    input_ids,
                    current_gamma,
                )
            )

            draft_texts = [
                self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                )
                for token_id in draft_tokens
            ]

            print(
                f"Draft proposals: {draft_texts}"
            )

            # ---------------------------------------------
            # STEP 2: TARGET VERIFICATION
            # ---------------------------------------------

            verification = self._verify_tokens(
                input_ids,
                draft_tokens,
                draft_logits,
                allow_bonus,
            )

            committed_tokens = verification[
                "committed_tokens"
            ]

            accepted_count = verification[
                "accepted_count"
            ]

            rejected_count = verification[
                "rejected_count"
            ]

            replacement_count = verification[
                "replacement_count"
            ]

            verified_count = verification[
                "verified_count"
            ]

            bonus_count = verification[
                "bonus_count"
            ]

            proposed_count = len(draft_tokens)

            # ---------------------------------------------
            # STEP 3: UPDATE METRICS
            # ---------------------------------------------

            run_metrics["rounds"] += 1

            run_metrics[
                "total_draft_tokens"
            ] += proposed_count

            run_metrics[
                "total_verified_draft_tokens"
            ] += verified_count

            run_metrics[
                "total_accepted_draft_tokens"
            ] += accepted_count

            run_metrics[
                "total_rejected_draft_tokens"
            ] += rejected_count

            run_metrics[
                "total_replacement_tokens"
            ] += replacement_count

            run_metrics[
                "total_bonus_target_tokens"
            ] += bonus_count

            run_metrics[
                "total_committed_tokens"
            ] += len(committed_tokens)

            if (
                accepted_count == proposed_count
                and rejected_count == 0
            ):
                run_metrics[
                    "full_accept_rounds"
                ] += 1

            print(
                f"Round {round_number} summary: "
                f"proposed={proposed_count}, "
                f"verified={verified_count}, "
                f"accepted={accepted_count}, "
                f"rejected={rejected_count}, "
                f"replacement={replacement_count}, "
                f"bonus={bonus_count}, "
                f"committed={len(committed_tokens)}"
            )

            committed_texts = [
                self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                )
                for token_id in committed_tokens
            ]

            print(
                f"Committed tokens: {committed_texts}"
            )

            # ---------------------------------------------
            # STEP 4: COMMIT VERIFIED OUTPUT
            # ---------------------------------------------

            committed_tensor = torch.tensor(
                [committed_tokens],
                device=self.device,
                dtype=input_ids.dtype,
            )

            input_ids = torch.cat(
                [input_ids, committed_tensor],
                dim=1,
            )

            num_generated += len(committed_tokens)

            if (
                self.tokenizer.eos_token_id
                in committed_tokens
            ):
                break

        end_time = time.perf_counter()

        actual_tokens_generated = (
            input_ids.shape[1] - original_input_len
        )

        total_time = max(
            end_time - start_time,
            1e-9,
        )

        speed = (
            actual_tokens_generated / total_time
        )

        proposed_tokens = run_metrics[
            "total_draft_tokens"
        ]

        if proposed_tokens > 0:
            run_metrics["acceptance_rate"] = (
                run_metrics[
                    "total_accepted_draft_tokens"
                ]
                / proposed_tokens
            )

        self.metrics = run_metrics

        print(
            "\nRun stats: "
            f"accepted draft tokens="
            f"{run_metrics['total_accepted_draft_tokens']}/"
            f"{proposed_tokens}, "
            f"acceptance rate="
            f"{run_metrics['acceptance_rate']:.2%}, "
            f"bonus target tokens="
            f"{run_metrics['total_bonus_target_tokens']}"
        )

        text = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True,
        )

        return speed, text