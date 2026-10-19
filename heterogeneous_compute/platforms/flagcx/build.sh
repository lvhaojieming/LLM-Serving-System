#!/usr/bin/env bash
# Run inside a retained vendor container. No system libraries are replaced.
set -euo pipefail
source_dir="$1"
backend="$2"
case "$backend" in
  musa) flag=USE_MUSA; device=/usr/local/musa; libdir=lib ;;
  iluvatar) flag=USE_ILUVATAR; device=/usr/local/corex; libdir=lib64 ;;
  *) echo "Unsupported backend: $backend" >&2; exit 2 ;;
esac
cd "$source_dir"
make "$flag=1" "DEVICE_HOME=$device" "CCL_HOME=$device" \
  "DEVICE_LIB=$device/$libdir" "CCL_LIB=$device/$libdir" \
  HOST_CXX_STANDARD=-std=c++17 -j8
g++ -std=c++17 -shared -fPIC \
  /workspace/heterogeneous_compute/platforms/flagcx/bridge.cc \
  -I "$source_dir/build/include" -L "$source_dir/build/lib" \
  -Wl,-rpath,"$source_dir/build/lib" -lflagcx \
  -o "$source_dir/build/lib/libheteroserve_flagcx.so"
sha256sum "$source_dir/build/lib/"*.so
echo FLAGCX_BUILD_COMPLETE
