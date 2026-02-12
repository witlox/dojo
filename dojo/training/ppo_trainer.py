"""PPO trainer with LoRA/QLoRA for behavioral fine-tuning."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrainerConfig:
    """Configuration for the PPO trainer."""

    base_model: str = "deepseek-ai/deepseek-coder-v2-lite-instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    quantization_bits: int = 4  # 4-bit QLoRA via bitsandbytes
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 4
    cliprange: float = 0.2
    max_grad_norm: float = 1.0
    output_dir: str = "/tmp/dojo-training"
    device: str = "auto"


@dataclass
class TrainStepResult:
    """Result from a single PPO training step."""

    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    num_samples: int
    metadata: Dict[str, float] = field(default_factory=dict)


class DojoTrainer:
    """PPO trainer with LoRA/QLoRA adapters for behavioral fine-tuning.

    Wraps HuggingFace TRL's PPOTrainer for the dojo-specific workflow:
    - Base model: DeepSeek Coder V2 16B (or smaller for testing)
    - LoRA on behavioral layers only (q_proj, v_proj)
    - 4-bit QLoRA quantization for ~32GB VRAM
    - Per-stage adapter checkpoints

    The actual model is served via vLLM for episode inference.
    Training happens on prompt/response/reward tuples extracted from episodes.
    """

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self.config = config or TrainerConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._ppo_trainer: Any = None
        self._initialized = False
        self._step_count = 0

    def initialize(self) -> None:
        """Load model and set up PPO trainer.

        This is expensive — call once, not per episode.
        """
        if self._initialized:
            return

        try:
            self._setup_model()
            self._initialized = True
        except ImportError as e:
            raise ImportError(
                f"Training dependencies not available: {e}. "
                "Ensure torch, transformers, peft, trl, bitsandbytes are installed."
            ) from e

    def _setup_model(self) -> None:
        """Set up the model, tokenizer, and PPO trainer."""
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

        cfg = self.config

        # Quantization config for QLoRA
        bnb_config = None
        if cfg.quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model with quantization
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map=cfg.device,
            trust_remote_code=True,
        )

        # LoRA config
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # PPO config
        ppo_config = PPOConfig(
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            mini_batch_size=cfg.mini_batch_size,
            ppo_epochs=cfg.ppo_epochs,
            cliprange=cfg.cliprange,
            max_grad_norm=cfg.max_grad_norm,
        )

        self._model = model
        self._ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=self._tokenizer,
        )

    def train_step(
        self,
        queries: List[str],
        responses: List[str],
        rewards: List[float],
    ) -> TrainStepResult:
        """Run a single PPO training step.

        Args:
            queries: Prompt strings (what the model received).
            responses: Response strings (what the model generated).
            rewards: Scalar rewards for each prompt/response pair.

        Returns:
            TrainStepResult with loss and metric breakdown.
        """
        if not self._initialized:
            self.initialize()

        import torch

        # Tokenize queries and responses
        query_tensors = [
            self._tokenizer.encode(q, return_tensors="pt").squeeze()
            for q in queries
        ]
        response_tensors = [
            self._tokenizer.encode(r, return_tensors="pt").squeeze()
            for r in responses
        ]
        reward_tensors = [torch.tensor(r) for r in rewards]

        # PPO step
        stats = self._ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        self._step_count += 1

        return TrainStepResult(
            loss=stats.get("ppo/loss/total", 0.0),
            policy_loss=stats.get("ppo/loss/policy", 0.0),
            value_loss=stats.get("ppo/loss/value", 0.0),
            entropy=stats.get("ppo/policy/entropy", 0.0),
            kl_divergence=stats.get("ppo/mean_non_score_reward", 0.0),
            num_samples=len(queries),
            metadata={k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
        )

    def save_adapter(self, name: str) -> Path:
        """Save current LoRA adapter to disk.

        Args:
            name: Adapter name (e.g., "lora-v1-stage1").

        Returns:
            Path to saved adapter directory.
        """
        if not self._initialized:
            raise RuntimeError("Trainer not initialized")

        output_path = Path(self.config.output_dir) / name
        output_path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(output_path))
        return output_path

    def load_adapter(self, path: str) -> None:
        """Load a previously saved LoRA adapter.

        Args:
            path: Path to the adapter directory.
        """
        if not self._initialized:
            self.initialize()

        from peft import PeftModel

        self._model = PeftModel.from_pretrained(self._model, path)

    @property
    def step_count(self) -> int:
        return self._step_count
