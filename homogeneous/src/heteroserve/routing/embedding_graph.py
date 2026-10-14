"""Warm, bounded shape graphs for a single unpadded causal Qwen3 prompt."""


class EmbeddingGraphs:
    def __init__(self, encoder, device, buckets):
        import torch
        import torch_npu
        if encoder.config.model_type != 'qwen3' or getattr(encoder, 'has_sliding_layers', False):
            raise ValueError('Embedding graphs require full causal Qwen3 attention')
        self.entries = {}
        self.hits = 0
        self.fallbacks = 0
        self.oversize = 0
        # Capture before readiness; a first request never pays capture latency.
        with torch.inference_mode():
            for size in buckets:
                ids = torch.zeros((1, size), dtype=torch.long, device=device)
                positions = torch.arange(size, device=device).unsqueeze(0)
                def forward():
                    # Same SDPA causal path as a single all-valid prompt.
                    # Right padding cannot influence preceding valid tokens.
                    return encoder(input_ids=ids, position_ids=positions,
                        attention_mask={'full_attention': None}, use_cache=False).last_hidden_state
                for _ in range(3):
                    forward()
                torch.npu.synchronize()
                graph = torch_npu.npu.NPUGraph()
                with torch_npu.npu.graph(graph):
                    hidden = forward()
                graph.replay()
                torch.npu.synchronize()
                self.entries[size] = (ids, positions, graph, hidden)

    def pooled(self, tokens):
        length = tokens.shape[1]
        if tokens.shape[0] != 1:
            raise ValueError('Embedding graphs serve exactly one prompt at a time')
        bucket = next((n for n in self.entries if n >= length), None)
        if bucket is None:
            self.oversize += 1
            return None
        ids, positions, graph, hidden = self.entries[bucket]
        ids[:, :length].copy_(tokens)
        graph.replay()
        self.hits += 1
        return hidden[:, length-1, :]
