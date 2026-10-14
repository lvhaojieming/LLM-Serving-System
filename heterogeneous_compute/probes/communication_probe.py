"""Probe cross-vendor PP transport without loading weights or changing services.

Four ranks map to TP groups [0,1]/[2,3] and PP groups [0,2]/[1,3].
CPU staging success is transport evidence only, not model inference acceptance.
"""
import argparse
import datetime
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--device', choices=['npu', 'musa'], required=True)
    parser.add_argument('--master', required=True)
    parser.add_argument('--port', type=int, default=29643)
    parser.add_argument('--local-master')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    if args.device == 'npu':
        import torch_npu  # noqa: F401
    else:
        import torch_musa  # noqa: F401
    import torch.distributed as dist
    report = dict(rank=args.rank, device=args.device, torch=torch.__version__,
                  registered_backends=list(dist.Backend.backend_list),
                  tp=2, pp=2, model_loaded=False,
                  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
    try:
        device = f'{args.device}:{args.rank % 2}'
        getattr(torch, args.device).set_device(args.rank % 2)
        if args.local_master:
            backend = 'hccl' if args.device == 'npu' else 'mccl'
            dist.init_process_group(backend,
                                    init_method=f'tcp://{args.local_master}:{args.port + 1}',
                                    rank=args.rank % 2, world_size=2,
                                    timeout=datetime.timedelta(seconds=20))
            local = torch.tensor([args.rank % 2 + 1], dtype=torch.float32, device=device)
            dist.all_reduce(local)
            assert local.cpu().item() == 3
            report['local_accelerator_tp2'] = dict(status='passed', backend=backend)
            dist.destroy_process_group()
        dist.init_process_group('gloo', init_method=f'tcp://{args.master}:{args.port}',
                                rank=args.rank, world_size=4,
                                timeout=datetime.timedelta(seconds=20))
        value = torch.tensor([args.rank], dtype=torch.int64)
        dist.all_reduce(value)
        assert value.item() == 6
        report['four_rank_cpu_collective'] = 'passed'
        # Actual Qwen3 hidden dimension; equal PP boundaries, not two HTTP calls.
        shape = (262, 5120)
        tensor = torch.full(shape, args.rank + 1, dtype=torch.bfloat16, device=device).cpu()
        peer = args.rank + 2 if args.rank < 2 else args.rank - 2
        start = time.perf_counter()
        if args.rank < 2:
            dist.send(tensor, peer)
            dist.recv(tensor, peer)
            assert torch.all(tensor == args.rank + 2)
        else:
            dist.recv(tensor, peer)
            received = tensor.to(device)
            assert torch.all(received.cpu() == peer + 1)
            tensor = received.add_(1).cpu()
            dist.send(tensor, peer)
        report['cpu_staged_pp'] = dict(status='passed', shape=shape,
                                       bytes=tensor.numel() * tensor.element_size(),
                                       elapsed_ms=1000 * (time.perf_counter() - start))
        dist.barrier()
        accelerator = tensor.to(device)
        try:
            if args.rank < 2:
                dist.send(accelerator, peer)
            else:
                dist.recv(accelerator, peer)
            report['direct_device_gloo_pp'] = dict(status='passed')
        except Exception as error:
            report['direct_device_gloo_pp'] = dict(status='failed', error=str(error))
    except Exception as error:
        report['fatal_error'] = str(error)
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as error:
                report['cleanup_error'] = str(error)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
