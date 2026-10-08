"""Use the existing adapter; enable the TP extension only for opted-in model processes."""
import os

if os.environ.get("MOQE_ASCEND_INT4_ADAPTER") == "1":
    try:
        from adapter_patch import install
        install()
        if os.environ.get("MOQE_ASCEND_INT4_PARALLEL") == "1":
            from native_parallel import install
            install()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
