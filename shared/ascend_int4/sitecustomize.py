"""Opt-in registration in the serving process and spawned vLLM workers."""
import os
if os.environ.get('MOQE_ASCEND_INT4_ADAPTER') == '1':
    try:
        import shared.ascend_int4.vllm_adapter
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
