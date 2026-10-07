"""Ascend inference for legacy and frozen contextual probability routers."""
import sys
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from .runtime import RouterSettings
from .runtime import RoutingDecision


class AscendRouterRuntime:
    def __init__(self, settings: RouterSettings):
        settings.verify_checkpoint()
        source = Path(settings.training_code) / "src"
        if not (source / "moqe_router/model.py").is_file():
            raise ValueError("training_code must contain the matching moqe_router architecture")
        sys.path.insert(0, str(source))
        import torch
        import torch_npu  # Registers the NPU backend.
        from transformers import AutoTokenizer
        from moqe_router.evaluation import load_router

        if not settings.device.startswith("npu:"):
            raise ValueError("Ascend runtime requires an npu device")
        self.torch = torch
        self.device = torch.device(settings.device)
        torch.npu.set_device(self.device)
        torch.backends.mha.set_fastpath_enabled(False)
        checkpoint = torch.load(settings.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture_version") == "frozen-contextual-expert-softmax-v2":
            from moqe_router.contextual import ContextualPreferenceHead
            from transformers import AutoModel
            ids = tuple(checkpoint["expert_ids"])
            if len(ids) != 2:
                raise ValueError("The migrated threshold policy requires exactly two experts")
            if set(settings.expert_mapping) != set(ids) or len(set(settings.expert_mapping.values())) != len(ids):
                raise ValueError("Expert mapping must match contextual checkpoint outputs")
            self.experts = tuple(settings.expert_mapping[id_] for id_ in ids)
            self.contextual = True
            self.threshold = checkpoint["threshold"]
            encoder_path = settings.embedding_model or checkpoint["settings"]["encoder"]
            self.max_prompt_tokens = checkpoint["settings"]["max_encoder_tokens"]
            if settings.embedding_graph and settings.graph_buckets[-1] > self.max_prompt_tokens:
                raise ValueError("Graph bucket exceeds contextual encoder token limit")
            self.router = ContextualPreferenceHead(checkpoint["feature_dim"], kind=checkpoint["kind"],
                hidden_dim=checkpoint["hidden_dim"], dropout=checkpoint["dropout"], num_experts=len(ids))
            self.router.load_state_dict(checkpoint["head_state_dict"])
            self.router.eval().requires_grad_(False)
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
            self.encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation=checkpoint["settings"]["attention"]).to(self.device)
            self.encoder.eval().requires_grad_(False)
            self.embedding_graphs = None
            self.graph_threshold_margin = settings.graph_threshold_margin
            if settings.embedding_graph:
                from .embedding_graph import EmbeddingGraphs
                self.embedding_graphs = EmbeddingGraphs(self.encoder, self.device, settings.graph_buckets)
            self.tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer, local_files_only=True)
            self.chat_template = self.tokenizer.get_chat_template()
            if not self.chat_template:
                raise ValueError("Canonical tokenizer must supply a chat template")
            self.lock = threading.Lock()
            self.route([{"role": "user", "content": "Hello"}], 128, {"enable_thinking": False})
            return
        if settings.embedding_graph or settings.embedding_model is not None:
            raise ValueError("Embedding graph/model settings require a contextual V7 checkpoint")
        self.contextual = False
        self.max_prompt_tokens = checkpoint["training_config"]["max_prompt_tokens"]
        ids = tuple(checkpoint["architecture_config"]["expert_ids"])
        if set(settings.expert_mapping) != set(ids):
            raise ValueError("Expert mapping must match every checkpoint expert id exactly")
        if len(set(settings.expert_mapping.values())) != len(ids):
            raise ValueError("Each checkpoint output must map to a distinct expert pool")
        self.experts = tuple(settings.expert_mapping[id_] for id_ in ids)
        del checkpoint
        self.router, self.embedding, _ = load_router(settings.checkpoint, self.device)
        self.router.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer, local_files_only=True)
        self.chat_template = self.tokenizer.get_chat_template()
        if not self.chat_template:
            raise ValueError("Canonical tokenizer must supply a chat template")
        self.lock = threading.Lock()
        self.route([{"role": "user", "content": "Hello"}], 128, {"enable_thinking": False})

    def route(self, messages, max_new_tokens, template_kwargs):
        with self.lock:
            torch = self.torch
            torch.npu.set_device(self.device)  # Device context is thread-local.
            started = time.perf_counter()
            ids = self.tokenizer.apply_chat_template(
                messages, chat_template=self.chat_template, tokenize=True,
                add_generation_prompt=True, return_dict=False, **template_kwargs,
            )
            if isinstance(ids, Mapping):
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if not ids or len(ids) > self.max_prompt_tokens:
                raise ValueError(f"Prompt must have 1..{self.max_prompt_tokens} tokens; no truncation")
            if self.contextual:
                from moqe_router.contextual import last_valid_pool, prompt_features
                text = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                encoded = self.encoder_tokenizer(text, truncation=False, return_tensors="pt")
                if encoded["input_ids"].shape[1] > self.max_prompt_tokens:
                    raise ValueError("Full prompt exceeds embedding context; no truncation")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.inference_mode():
                    def predict(pooled):
                        features = prompt_features(pooled, torch.tensor([len(ids)], device=self.device),
                            torch.tensor([max_new_tokens], device=self.device)).cpu()
                        return self.router(features).float().softmax(-1)[0].tolist()
                    pooled = self.embedding_graphs.pooled(encoded['input_ids']) if self.embedding_graphs else None
                    used_graph = pooled is not None
                    if not used_graph:
                        hidden = self.encoder(**encoded, use_cache=False).last_hidden_state
                        pooled = last_valid_pool(hidden, encoded["attention_mask"])
                    probabilities = predict(pooled)
                    # Bucket padding can change BF16 rounding. Use original
                    # execution near the expert decision boundary.
                    if used_graph and abs(probabilities[1]-self.threshold) <= self.graph_threshold_margin:
                        self.embedding_graphs.fallbacks += 1
                        hidden = self.encoder(**encoded, use_cache=False).last_hidden_state
                        probabilities = predict(last_valid_pool(hidden, encoded['attention_mask']))
                    if not all(math.isfinite(p) for p in probabilities):
                        raise RuntimeError("Router produced nonfinite probabilities")
                index = int(probabilities[1] > self.threshold)
                return RoutingDecision(self.experts[index], dict(zip(self.experts, probabilities)),
                    len(ids), (time.perf_counter() - started) * 1000)
            tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
            mask = torch.ones_like(tokens, dtype=torch.bool)
            generation = torch.tensor([max_new_tokens], dtype=torch.long, device=self.device)
            with torch.inference_mode(), torch.autocast("npu", dtype=torch.bfloat16):
                logits = self.router(self.embedding(tokens), mask, generation)
                if not bool(torch.isfinite(logits).all().cpu()):
                    raise RuntimeError("Router produced nonfinite logits")
                probabilities = logits.float().softmax(-1).cpu()[0].tolist()
            index = max(range(len(probabilities)), key=probabilities.__getitem__)
            return RoutingDecision(self.experts[index], dict(zip(self.experts, probabilities)),
                                   len(ids), (time.perf_counter() - started) * 1000)
