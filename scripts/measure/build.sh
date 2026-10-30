#!/bin/bash
# build.sh SRC BLD PKG [extra cmake args...]  -> PKG/imfeat/{*.py,*.so}
set -e
SRC=$1; BLD=$2; PKG=$3; shift 3
HWY=${IMFEAT_HIGHWAY_SRC:?set IMFEAT_HIGHWAY_SRC to a Highway 1.2.0 checkout}
cmake -S "$SRC" -B "$BLD" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=$(which python3) \
  -Dnanobind_DIR=$(python3 -m nanobind --cmake_dir) -DFETCHCONTENT_SOURCE_DIR_HIGHWAY=$HWY "$@" > "$BLD.cfg.log" 2>&1
cmake --build "$BLD" -j ${BUILD_JOBS:-8} > "$BLD.build.log" 2>&1
rm -rf "$PKG"; mkdir -p "$PKG/imfeat"
cp "$SRC"/src/imfeat/*.py "$PKG/imfeat/"; cp "$SRC"/src/imfeat/*.typed "$PKG/imfeat/" 2>/dev/null || true
cp "$BLD"/imfeat_core*.so "$PKG/imfeat/"
md5sum "$PKG"/imfeat/*.so
