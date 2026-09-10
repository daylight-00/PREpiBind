# A prebuilt AlphaFold 3 wheel

`pixi install` here builds `alphafold3` from source, which needs a C++ toolchain, CMake >= 3.28 and
network access at configure time. If that is inconvenient, the same commit is available as a wheel.

    alphafold3-3.0.1-cp311-cp311-linux_x86_64.whl
    241.7 MB
    sha256  8911bb4053c1992ae8e69ee2edb2d651d469b94cf9f17c79d237c1e86c4a0680
    built from  github.com/daylight-00/alphafold3 @ f1b872e7375d82b322cde5f2ac5efa2cda33f1f4
                clean checkout, no working-tree modifications

It is **not** part of the specification. A wheel with CPython ABI tags and a Linux x86-64 platform
tag only installs on that combination; the `pyproject.toml` and `pixi.lock` beside it are what
define the environment.

The wheel already bundles `share/libcifpp/components.cif`, so the `prepare` task (`build_data`) is
not needed after installing it.

## What the build needs, learned by hitting each one

Two failures on the way, both worth knowing before anyone repeats this:

- **zlib headers.** libcifpp includes `zlib.h`. Putting the conda prefix on `PATH` is not enough;
  `CPATH` and `CMAKE_PREFIX_PATH` have to point at it too.
- **Do not export `LD_LIBRARY_PATH`.** The conda OpenSSL shadows the system one, and
  `git-remote-https` then fails with `undefined symbol: EVP_md2`. CMake FetchContent clones
  abseil-cpp, pybind11, pybind11_abseil, libcifpp and dssp over HTTPS, so the whole configure step
  dies. `CPATH` and `LIBRARY_PATH` affect compilation only and are safe.

Build recipe:

    git clone https://github.com/daylight-00/alphafold3 && git checkout f1b872e7
    export PATH=<af3-env>/bin:$PATH CC=<af3-env>/bin/gcc CXX=<af3-env>/bin/g++
    export CPATH=<af3-env>/include LIBRARY_PATH=<af3-env>/lib CMAKE_PREFIX_PATH=<af3-env>
    pip install "cmake>=3.28" ninja scikit-build-core pybind11 numpy
    pip wheel . --no-deps --no-build-isolation -w dist

Roughly 235 compilation units; a few minutes on 16 CPU cores. No GPU needed.
