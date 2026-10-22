"""The adapter's diagnostic code must stay outside a full compiled forward."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def test_int4_forward_compiles_without_logging_or_counter_mutation():
    # Load the production forward body without importing a vendor runtime.
    source = Path(__file__).resolve().parents[2] / "shared/ascend_int4/vllm_adapter.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendInt4LinearMethod")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply")
    scope = {"torch": torch, "json": json,
             "matmul": lambda x, q, scales, zeros, group_size, bias: x @ q}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    adapter = SimpleNamespace(quant_config=SimpleNamespace(kind="awq", group_size=128), calls=0)
    layer = SimpleNamespace(qweight=torch.eye(2), scales=torch.ones(1, 2), qzeros=torch.zeros(1, 2))
    forward = torch.compile(lambda x: scope["apply"](adapter, layer, x), backend="eager", fullgraph=True)
    x = torch.tensor([[1., 2.]])
    torch.testing.assert_close(forward(x), x)
    torch.testing.assert_close(forward(x + 1), x + 1)
    assert adapter.calls == 0
