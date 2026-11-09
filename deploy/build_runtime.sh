#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../go2_runtime"
sdk="${UNITREE_SDK2_DIR:-/home/lenovo/unitree_sdk2}"
mkdir -p build
g++ -std=c++17 -O3 -shared -fPIC bev_reduce.cpp -o build/libbev_reduce.so.new
mv -f build/libbev_reduce.so.new build/libbev_reduce.so
arch="$(uname -m)"
case "$arch" in x86_64|aarch64) ;; *) echo "Unsupported architecture: $arch" >&2; exit 2;; esac
g++ -std=c++17 -O2 -pthread -I"$sdk/include" -I"$sdk/thirdparty/include" \
  -I"$sdk/thirdparty/include/ddscxx" robot_bridge.cpp "$sdk/lib/$arch/libunitree_sdk2.a" \
  -L"$sdk/thirdparty/lib/$arch" -Wl,-rpath,"$sdk/thirdparty/lib/$arch" \
  -lddsc -lddscxx -o build/robot_bridge.new
mv -f build/robot_bridge.new build/robot_bridge
