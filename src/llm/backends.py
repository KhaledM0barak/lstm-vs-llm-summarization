"""Generation backends for the LLM baseline.

Two backends produce identical record formats so the evaluation harness and the
report tables are backend-agnostic:

  `mlx`       - a locally run open-weights instruction-tuned model on Apple
                silicon via MLX. Free; cost is reported as wall-clock GPU time
                rather than USD.
  `anthropic` - a hosted low-cost API model. Cost is reported in USD from the
                per-request token usage the API returns.

The assignment permits either. Both are implemented so the same prompts,
resumability, and accounting apply whichever is used.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class GenResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    error: str | None = None


@dataclass
class BackendInfo:
    """Everything the report needs to describe how the baseline was run."""

    name: str
    model: str
    kind: str                       # "local" or "api"
    details: dict = field(default_factory=dict)


class AnthropicBackend:
    """Hosted API. Requests are independent, so they are issued concurrently."""

    kind = "api"
    # USD per million tokens for the configured model.
    PRICE_PER_MTOK_IN = 1.00
    PRICE_PER_MTOK_OUT = 5.00

    def __init__(self, model: str = "claude-haiku-4-5", max_retries: int = 5) -> None:
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic(max_retries=max_retries)

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="anthropic",
            model=self.model,
            kind=self.kind,
            details={
                "price_per_mtok_input_usd": self.PRICE_PER_MTOK_IN,
                "price_per_mtok_output_usd": self.PRICE_PER_MTOK_OUT,
            },
        )

    def supports_concurrency(self) -> bool:
        return True

    def generate_one(self, system: str, messages: list[dict], max_tokens: int) -> GenResult:
        t0 = time.time()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced in the run summary
            return GenResult("", 0, 0, time.time() - t0, f"{type(exc).__name__}: {exc}")

        text = "".join(b.text for b in resp.content if b.type == "text")
        return GenResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=time.time() - t0,
        )

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1e6 * self.PRICE_PER_MTOK_IN
            + output_tokens / 1e6 * self.PRICE_PER_MTOK_OUT
        )


class MLXBackend:
    """Locally run open-weights model on Apple silicon via MLX.

    Generation is batched: prompts are padded and decoded together in one GPU
    pass, which is far better utilization than one request at a time. There is a
    single GPU and MLX state is not thread-safe, so the caller must not run this
    backend concurrently -- batching is the parallelism.
    """

    kind = "local"

    def __init__(
        self,
        model: str = "mlx-community/Llama-3.1-8B-Instruct-4bit",
        batch_size: int = 8,
        temperature: float = 0.0,
    ) -> None:
        # transformers emits a 400-character advisory about
        # clean_up_tokenization_spaces every time this tokenizer is constructed.
        # It is not actionable -- mlx-lm owns that call -- and it lands in the
        # middle of the recorded demo, three times, wrapping over a dozen lines.
        import os
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        # huggingface_hub prints "Fetching 6 files: 100%|..." plus reconstruction
        # and download bars on every load, even when nothing is downloaded. That
        # is three blocks of progress noise in the middle of a recorded demo.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        self.model_name = model
        self.batch_size = batch_size
        self.temperature = temperature
        print(f"loading {model} (first run downloads ~4.5 GB) ...")
        t0 = time.time()
        self.model, self.tokenizer = load(model)
        print(f"loaded in {time.time() - t0:.1f}s")
        # Greedy decoding: the baseline should be deterministic and reproducible.
        self.sampler = make_sampler(temp=temperature)
        self._lock = threading.Lock()

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="mlx",
            model=self.model_name,
            kind=self.kind,
            details={
                "quantization": "4-bit",
                "batch_size": self.batch_size,
                "sampling": f"greedy (temp={self.temperature})",
                "framework": "mlx-lm",
            },
        )

    def supports_concurrency(self) -> bool:
        return False

    def _encode(self, system: str, messages: list[dict]) -> list[int]:
        chat = [{"role": "system", "content": system}] + messages
        ids = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        # Some tokenizer wrappers return a string; normalize to token ids.
        if isinstance(ids, str):
            ids = self.tokenizer.encode(ids)
        return list(ids)

    def generate_batch(
        self,
        system: str,
        message_lists: list[list[dict]],
        max_tokens: int,
    ) -> list[GenResult]:
        from mlx_lm import batch_generate

        prompts = [self._encode(system, m) for m in message_lists]
        t0 = time.time()
        with self._lock:
            resp = batch_generate(
                self.model,
                self.tokenizer,
                prompts=prompts,
                max_tokens=max_tokens,
                sampler=self.sampler,
                verbose=False,
            )
        elapsed = time.time() - t0
        # Wall-clock is shared across the batch; attribute it evenly so the
        # reported per-summary latency reflects real throughput.
        per_item = elapsed / max(len(prompts), 1)

        results = []
        for ids, text in zip(prompts, resp.texts):
            out_tokens = len(self.tokenizer.encode(text)) if text else 0
            results.append(
                GenResult(
                    text=text,
                    input_tokens=len(ids),
                    output_tokens=out_tokens,
                    latency_s=per_item,
                )
            )
        return results

    def generate_one(self, system: str, messages: list[dict], max_tokens: int) -> GenResult:
        return self.generate_batch(system, [messages], max_tokens)[0]

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0  # Local inference: cost is reported as GPU-hours instead.


def build_backend(name: str, model: str | None, batch_size: int):
    if name == "anthropic":
        return AnthropicBackend(model or "claude-haiku-4-5")
    if name == "mlx":
        return MLXBackend(
            model or "mlx-community/Llama-3.1-8B-Instruct-4bit",
            batch_size=batch_size,
        )
    raise ValueError(f"unknown backend {name!r}")
