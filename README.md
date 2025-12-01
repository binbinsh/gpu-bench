# gpu-bench
Multi-GPU CUDA quantized compute/memory benchmark suite (fp4/fp8/bf16/fp16/fp32/fp64). Follows CUDA sample patterns (`matrixMulCUBLAS`, `bandwidthTest`) to hit 100% SM and memory load.

Local benchmark results (cuBLASLt + Transformer Engine), run on my desktop with `matrix=8192`, `compute-iters=30`, `seconds=120` (FP4 uses `seconds=180`):

| Metric | [RTX 5090 D](results/5090d.json) | [RTX 4090](results/4090.json) |
| --- | --- | --- |
| SM | 12.0 | 8.9 |
| fp32 (TFLOP/s) | 116.3 | 89.9 |
| bf16 (TFLOP/s) | 234.8 | 175.1 |
| fp16 (TFLOP/s) | 236.3 | 175.1 |
| fp8 with fp32 accum (TFLOP/s) | 548.5 | N/A |
| fp4 with fp32 accum (TFLOP/s) | 1040.0 | N/A |
| Memory D2D memcpy (GB/s) | 1528.8 | 918.8 |
| Memory Streaming (GB/s) | 1340.8 | 842.7 |

**Note:** FP8 tensor cores require SM90+ (Hopper/Blackwell). RTX 4090 (Ada, SM89) does not expose FP8 kernels; NVIDIA forum confirmation: https://forums.developer.nvidia.com/t/4090-doesnt-have-fp8-compute/232256


## Quick start
```bash
git clone https://github.com/binbinsh/gpu-bench.git
cd gpu-bench
make            # builds gpu_bench (defaults to sm_89)
```
Choose an arch: `make COMPUTE=90` (Ada/Hopper), `make COMPUTE=100` or `make COMPUTE=120` for newer Blackwell parts when available. Clean artifacts: `make clean`.

FP4 path (CUDA >= 12.8 with TransformerEngine):
```bash
git submodule update --init --recursive transformer-engine
CXX=/usr/bin/g++-12 CC=/usr/bin/gcc-12 cmake -S transformer-engine/transformer_engine/common -B transformer-engine/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build transformer-engine/build -j32
make COMPUTE=89,120 ENABLE_FP4=1 CCPATH=/usr/bin/g++-12 TE_LIBDIR=transformer-engine/build TE_INCLUDEDIR=transformer-engine/transformer_engine/common/include
```
Tip: `COMPUTE` accepts comma-separated arch codes for fatbin builds, e.g. `COMPUTE=89,120` to target both Ada (sm_89) and Blackwell (sm_120).

If you previously built Transformer Engine with a different compiler, clear the cache first:
```bash
rm -rf transformer-engine/build
```

## Running
List GPUs: `./gpu_bench --list`

Quantized compute + memory sweep (all GPUs, steady 2 min):
```bash
./gpu_bench --matrix 8192 --compute-iters 30 --seconds 60 --parallel --report results/all.json
```

B200/Blackwell focus with tensor cores:
```bash
./gpu_bench --devices 0 --precisions bf16,fp16,fp8,fp32 --matrix 8192 --compute-iters 30 --no-memory --seconds 60 --report results/b200.json
```

FP4 check (Transformer Engine, 5090/B200-class):
```bash
./gpu_bench --devices 1 --precisions fp4 --matrix 8192 --compute-iters 80 --seconds 60 --no-memory --report results/fp4.json
```

Memory-only sweep:
```bash
./gpu_bench --no-compute --mem 40% --mem-iters 50 --seconds 60 --report results/mem.json
```

Key flags:
- `--precisions`: `fp4,fp8,fp16,bf16,fp32,fp64`
- `--compute-iters`: number of GEMM iterations per timing loop
- `--seconds`: keep each phase running for at least N seconds
- `--devices`: comma-separated CUDA device indices to run (e.g., `--devices 1`)
- `--parallel`: run all selected GPUs concurrently
- `--no-memory`: skip memory bandwidth test (compute-only)
- `--no-compute`: skip GEMM (memory-only)
- `--mem <bytes|percent%>`: control bandwidth test size
- `--mem-iters <N>`: control bandwidth test length
- `--report <path>`: JSON summary per GPU

## Acknowledgments
- Inspired by the [gpu-burn](https://github.com/wilicc/gpu-burn) project.
