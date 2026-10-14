"""Use the existing adapter; enable the TP extension only for opted-in model processes."""
import os

if os.environ.get("MOQE_ASCEND_INT4_ADAPTER") == "1":
    try:
        from adapter_patch import install
        install()
        if os.environ.get("MOQE_ASCEND_INT4_PARALLEL") == "1":
            from native_parallel import install
            fast_path = os.environ.get("MOQE_ASCEND_INT4_FAST_PATH")
            install(fast_path=None if fast_path is None else fast_path == "1")
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(78)
