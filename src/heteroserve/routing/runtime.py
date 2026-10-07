from dataclasses import dataclass
from typing import Protocol
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class RouterSettings:
    checkpoint: str
    tokenizer: str
    training_code: str
    expert_mapping: dict[str, str]
    checkpoint_sha256: str
    device: str = "npu:0"
    embedding_graph: bool = False
    graph_buckets: tuple[int, ...] = (64, 128, 256, 512, 1024)
    graph_threshold_margin: float = 0.01
    embedding_model: str | None = None

    def __post_init__(self):
        if not self.expert_mapping or len(set(self.expert_mapping.values())) != len(self.expert_mapping):
            raise ValueError("Checkpoint outputs must map to distinct expert pools")
        if not self.device.startswith("npu:"):
            raise ValueError("Only Ascend learned routing is supported")
        if len(self.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.checkpoint_sha256):
            raise ValueError("Router checkpoint must have a pinned SHA-256")
        if type(self.embedding_graph) is not bool or not self.graph_buckets or any(type(n) is not int or n < 2 for n in self.graph_buckets):
            raise ValueError("Invalid embedding graph configuration")
        if list(self.graph_buckets) != sorted(set(self.graph_buckets)) or not 0 <= self.graph_threshold_margin < .5:
            raise ValueError("Invalid graph buckets or threshold margin")

    def verify_checkpoint(self):
        digest = hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest()
        if digest != self.checkpoint_sha256:
            raise ValueError("Router checkpoint checksum mismatch")


@dataclass(frozen=True)
class RoutingDecision:
    expert: str
    probabilities: dict[str, float]
    input_tokens: int
    elapsed_ms: float


class RouterRuntime(Protocol):
    chat_template: str

    def route(self, messages: list[dict], max_new_tokens: int, template_kwargs: dict) -> RoutingDecision: ...
