import torch
from tqdm import tqdm

from src.losses.bradley_terry import bradley_terry_loss
from src.evaluation.evaluator import evaluate


class RewardModelTrainer:

    def __init__(
        self,
        model,
        optimizer,
        device,
        scheduler=None,
        gradient_clip_norm=None,
        mixed_precision=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm

        # ---------------------------------------------------------
        # Mixed precision configuration
        # ---------------------------------------------------------

        self.mixed_precision = mixed_precision

        self.use_bf16 = (
            mixed_precision == "bf16"
            and device.type == "cuda"
        )

        if self.use_bf16:

            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(
                    "BF16 was requested, but this GPU "
                    "does not support BF16."
                )

            print("Mixed precision: BF16")

        else:
            print("Mixed precision: disabled")

    # =============================================================
    # TRAINING
    # =============================================================

    def train_epoch(
        self,
        dataloader,
        global_step,
        epoch=None,
        total_epochs=None,
    ):

        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        # ---------------------------------------------------------
        # Progress bar description
        # ---------------------------------------------------------

        if epoch is not None and total_epochs is not None:
            description = (
                f"Epoch {epoch}/{total_epochs}"
            )
        else:
            description = "Training"

        progress_bar = tqdm(
            dataloader,
            desc=description,
            unit="batch",
        )

        # =========================================================
        # BATCH LOOP
        # =========================================================

        for batch in progress_bar:

            # -----------------------------------------------------
            # Move batch to GPU
            # -----------------------------------------------------

            batch = {
                key: value.to(self.device)
                for key, value in batch.items()
            }

            # -----------------------------------------------------
            # Clear previous gradients
            # -----------------------------------------------------

            self.optimizer.zero_grad(
                set_to_none=True
            )

            # =====================================================
            # FORWARD PASS
            # =====================================================

            # BF16 is used for the model forward pass.
            #
            # We calculate the Bradley-Terry loss using FP32
            # rewards for a little extra numerical safety.
            #
            # BF16 does NOT require GradScaler.
            # =====================================================

            if self.use_bf16:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):

                    chosen_rewards = self.model(
                        input_ids=batch[
                            "chosen_input_ids"
                        ],
                        attention_mask=batch[
                            "chosen_attention_mask"
                        ],
                    )

                    rejected_rewards = self.model(
                        input_ids=batch[
                            "rejected_input_ids"
                        ],
                        attention_mask=batch[
                            "rejected_attention_mask"
                        ],
                    )

                # -------------------------------------------------
                # Calculate loss in FP32
                # -------------------------------------------------

                loss = bradley_terry_loss(
                    chosen_rewards=chosen_rewards.float(),
                    rejected_rewards=rejected_rewards.float(),
                )

            else:

                chosen_rewards = self.model(
                    input_ids=batch[
                        "chosen_input_ids"
                    ],
                    attention_mask=batch[
                        "chosen_attention_mask"
                    ],
                )

                rejected_rewards = self.model(
                    input_ids=batch[
                        "rejected_input_ids"
                    ],
                    attention_mask=batch[
                        "rejected_attention_mask"
                    ],
                )

                loss = bradley_terry_loss(
                    chosen_rewards=chosen_rewards,
                    rejected_rewards=rejected_rewards,
                )

            # =====================================================
            # BACKWARD PASS
            # =====================================================

            loss.backward()

            # -----------------------------------------------------
            # Gradient clipping
            # -----------------------------------------------------

            if self.gradient_clip_norm is not None:

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_norm,
                )

            # -----------------------------------------------------
            # Optimizer update
            # -----------------------------------------------------

            self.optimizer.step()

            # -----------------------------------------------------
            # Scheduler update
            # -----------------------------------------------------

            if self.scheduler is not None:
                self.scheduler.step()

            global_step += 1

            # =====================================================
            # METRICS
            # =====================================================

            batch_size = chosen_rewards.size(0)

            correct = (
                chosen_rewards > rejected_rewards
            ).sum().item()

            total_correct += correct
            total_examples += batch_size

            total_loss += (
                loss.item() * batch_size
            )

            average_loss = (
                total_loss / total_examples
            )

            accuracy = (
                total_correct / total_examples
            )

            # -----------------------------------------------------
            # Current learning rate
            # -----------------------------------------------------

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            # =====================================================
            # GPU MEMORY
            # =====================================================

            if torch.cuda.is_available():

                gpu_memory_allocated = (
                    torch.cuda.memory_allocated(
                        self.device
                    )
                    / (1024 ** 3)
                )

                gpu_memory_reserved = (
                    torch.cuda.memory_reserved(
                        self.device
                    )
                    / (1024 ** 3)
                )

                gpu_memory_text = (
                    f"GPU: "
                    f"{gpu_memory_allocated:.1f}GB/"
                    f"{gpu_memory_reserved:.1f}GB"
                )

            else:

                gpu_memory_text = "GPU: N/A"

            # =====================================================
            # PROGRESS BAR
            # =====================================================

            progress_bar.set_postfix(
                loss=f"{average_loss:.4f}",
                acc=f"{accuracy:.4f}",
                lr=f"{current_lr:.2e}",
                gpu=gpu_memory_text,
                step=global_step,
            )

        progress_bar.close()

        # =========================================================
        # EPOCH METRICS
        # =========================================================

        average_loss = (
            total_loss / total_examples
        )

        accuracy = (
            total_correct / total_examples
        )

        return {
            "loss": average_loss,
            "accuracy": accuracy,
            "global_step": global_step,
        }

    # =============================================================
    # VALIDATION
    # =============================================================

    def validate(self, dataloader):

        return evaluate(
            model=self.model,
            dataloader=dataloader,
            device=self.device,
            mixed_precision=self.mixed_precision,
        )