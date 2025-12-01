#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#ifndef CUBLASLT_MATMUL_DESC_FAST_ACCUMULATION_MODE
#define CUBLASLT_MATMUL_DESC_FAST_ACCUMULATION_MODE CUBLASLT_MATMUL_DESC_FAST_ACCUM
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef ENABLE_FP4_BENCH
#include <cuda_fp4.h>
#include "transformer_engine/common/common.h"
#include "transformer_engine/common/include/transformer_engine/gemm.h"
#include "transformer_engine/common/include/transformer_engine/transformer_engine.h"
#endif

namespace {

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            std::ostringstream _oss;                                          \
            _oss << "CUDA error " << cudaGetErrorString(_err) << " at "       \
                 << __FILE__ << ":" << __LINE__;                              \
            throw std::runtime_error(_oss.str());                             \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(expr)                                                    \
    do {                                                                      \
        cublasStatus_t _err = (expr);                                         \
        if (_err != CUBLAS_STATUS_SUCCESS) {                                  \
            std::ostringstream _oss;                                          \
            _oss << "cuBLAS error " << _err << " at " << __FILE__ << ":"      \
                 << __LINE__;                                                 \
            throw std::runtime_error(_oss.str());                             \
        }                                                                     \
    } while (0)

// Precision choices used for quantization sweeps.
enum class Precision {
    kFP32,
    kFP16,
    kBF16,
    kFP8,
    kFP4,
    kFP64,
};

struct ComputeResult {
    Precision precision;
    size_t matrix_dim;
    int iterations;
    bool tensor_cores;
    double throughput;           // TFLOP/s
    const char *throughput_unit; // always TFLOP/s for listed precisions
    double elapsed_ms;
};

struct MemoryResult {
    size_t bytes;
    int d2d_iterations;
    int streaming_iterations;
    double d2d_bandwidth_gbs;
    double streaming_bandwidth_gbs;
};

struct DeviceReport {
    int device_index;
    std::string name;
    int sm_major;
    int sm_minor;
    size_t total_mem;
    size_t free_mem;
    std::vector<ComputeResult> compute;
    std::optional<MemoryResult> memory;
};

struct Options {
    bool list = false;
    bool run_compute = true;
    bool run_memory = true;
    bool tensor_cores = true;
    size_t matrix_dim = 8192;
    int compute_iters = 20;
    int warmup_iters = 2;
    int mem_iters = 30;
    long long mem_bytes = 0; // if zero, fall back to mem_percent
    int mem_percent = 90;
    std::vector<int> devices;
    std::vector<Precision> precisions{
        Precision::kFP32, Precision::kBF16, Precision::kFP16, Precision::kFP8};
    std::string report_path;
    double target_seconds = 0.0; // minimum time to keep each phase busy; 0 means single pass
    bool parallel = false;
};

const char *toString(Precision p) {
    switch (p) {
    case Precision::kFP32:
        return "fp32";
    case Precision::kFP16:
        return "fp16";
    case Precision::kBF16:
        return "bf16";
    case Precision::kFP8:
        return "fp8";
    case Precision::kFP4:
        return "fp4";
    case Precision::kFP64:
        return "fp64";
    }
    return "unknown";
}

cudaDataType_t dataType(Precision p) {
    switch (p) {
    case Precision::kFP32:
        return CUDA_R_32F;
    case Precision::kFP16:
        return CUDA_R_16F;
    case Precision::kBF16:
        return CUDA_R_16BF;
    case Precision::kFP8:
        return CUDA_R_8F_E4M3;
    case Precision::kFP4:
        return static_cast<cudaDataType_t>(0); // not used directly; FP4 handled via TE
    case Precision::kFP64:
        return CUDA_R_64F;
    }
    return CUDA_R_32F;
}

size_t elementSize(Precision p) {
    switch (p) {
    case Precision::kFP32:
        return sizeof(float);
    case Precision::kFP16:
        return sizeof(__half);
    case Precision::kBF16:
        return sizeof(__nv_bfloat16);
    case Precision::kFP8:
        return sizeof(__nv_fp8_e4m3);
    case Precision::kFP4:
        return 1; // packed nibble handled by TE
    case Precision::kFP64:
        return sizeof(double);
    }
    return sizeof(float);
}

bool usesTensorCore(Precision p) {
    return p == Precision::kFP16 || p == Precision::kBF16 || p == Precision::kFP8 ||
           p == Precision::kFP32 || p == Precision::kFP4;
}

std::vector<std::string> split(const std::string &input, char delim) {
    std::vector<std::string> out;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty())
            out.push_back(item);
    }
    return out;
}

Precision parsePrecision(const std::string &token) {
    if (token == "fp32")
        return Precision::kFP32;
    if (token == "fp16")
        return Precision::kFP16;
    if (token == "bf16")
        return Precision::kBF16;
    if (token == "fp8")
        return Precision::kFP8;
    if (token == "fp4")
        return Precision::kFP4;
    if (token == "fp64" || token == "double")
        return Precision::kFP64;
    throw std::runtime_error("Unknown precision: " + token);
}

std::vector<int> parseDeviceList(const std::string &csv) {
    std::vector<int> out;
    for (const auto &item : split(csv, ',')) {
        if (item == "all" || item == "ALL")
            continue;
        out.push_back(std::stoi(item));
    }
    return out;
}

bool parseMemoryArg(const std::string &value, Options &opts) {
    if (value.empty())
        return false;
    if (value.back() == '%') {
        opts.mem_percent = std::stoi(value.substr(0, value.size() - 1));
        opts.mem_bytes = 0;
        return true;
    }

    char suffix = value.back();
    long long factor = 1;
    std::string numeric = value;
    if (suffix == 'g' || suffix == 'G') {
        factor = 1024ll * 1024ll * 1024ll;
        numeric = value.substr(0, value.size() - 1);
    } else if (suffix == 'm' || suffix == 'M') {
        factor = 1024ll * 1024ll;
        numeric = value.substr(0, value.size() - 1);
    }
    opts.mem_bytes = std::stoll(numeric) * factor;
    opts.mem_percent = 0;
    return true;
}

void printHelp() {
    std::cout << "gpu_bench — quantized compute + memory full-load benchmark\n";
    std::cout << "Usage: ./gpu_bench [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --list                   List CUDA GPUs and exit\n";
    std::cout << "  --devices <csv|all>      Device indices, default=all\n";
    std::cout << "  --precisions <csv>       fp32,fp16,bf16,fp8,fp4,fp64 (default fp32,bf16,fp16,fp8)\n";
    std::cout << "  --matrix <N>             Square GEMM size (default 8192)\n";
    std::cout << "  --compute-iters <N>      GEMM iterations (default 20)\n";
    std::cout << "  --mem-iters <N>          Memory loop iterations (default 30)\n";
    std::cout << "  --seconds|--duration <N> Keep each compute/memory phase busy for >= N seconds\n";
    std::cout << "  --mem <bytes|percent%>   Device memory for bandwidth test (default 90%)\n";
    std::cout << "  --parallel               Run all selected GPUs concurrently\n";
    std::cout << "  --no-compute             Skip GEMM benchmarks\n";
    std::cout << "  --no-memory              Skip bandwidth benchmarks\n";
    std::cout << "  --no-tensor-cores        Disable tensor-core math modes\n";
    std::cout << "  --report <file>          Write JSON report\n";
    std::cout << "  --help                   Show this help\n";
    std::cout << "Notes:\n";
    std::cout << "  fp4 support requires building with ENABLE_FP4_BENCH=1 and a CUDA >= 12.8 toolkit plus Transformer Engine.\n";
}

template <typename T>
void fillData(std::vector<T> &buffer, Precision /*p*/) {
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = static_cast<T>((i % 13) + 1);
    }
}

template <>
void fillData(std::vector<__half> &buffer, Precision /*p*/) {
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = __float2half(static_cast<float>((i % 13) + 1));
    }
}

template <>
void fillData(std::vector<__nv_bfloat16> &buffer, Precision /*p*/) {
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = __float2bfloat16(static_cast<float>((i % 13) + 1));
    }
}

template <>
void fillData(std::vector<__nv_fp8_e4m3> &buffer, Precision /*p*/) {
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = __nv_fp8_e4m3(static_cast<float>((i % 13) + 1));
    }
}

cublasComputeType_t computeType(Precision p, bool tensor_cores) {
    switch (p) {
    case Precision::kFP32:
        return tensor_cores ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    case Precision::kFP16:
    case Precision::kBF16:
        return CUBLAS_COMPUTE_32F;
    case Precision::kFP8:
        return CUBLAS_COMPUTE_16F; // FP16 accumulate
    case Precision::kFP4:
        return CUBLAS_COMPUTE_32F;
    case Precision::kFP64:
        return CUBLAS_COMPUTE_64F;
    }
    return CUBLAS_COMPUTE_32F;
}

size_t clampMatrixDim(size_t requested, Precision p, double safety_ratio,
                      size_t free_mem_bytes) {
    size_t bytes_per_element = elementSize(p);
    // Three matrices (A, B, C) at the chosen precision. FP8 writes fp16 outputs.
    size_t output_bytes = (p == Precision::kFP8) ? sizeof(__half) : bytes_per_element;
    double usable_bytes = static_cast<double>(free_mem_bytes) * safety_ratio;
    double max_elems = usable_bytes / static_cast<double>(bytes_per_element * 2 + output_bytes);
    size_t max_dim = static_cast<size_t>(std::sqrt(max_elems));
    return std::max<size_t>(1, std::min(requested, max_dim));
}

#ifdef ENABLE_FP4_BENCH
constexpr size_t kFp4WorkspaceMinBytes = 64ull * 1024ull * 1024ull; // baseline for cuBLASLt FP4

size_t fp4WorkspaceBytes(size_t free_mem_bytes) {
    // Allow up to 20% of free memory for workspace, capped to 1GB, but never below 64MB.
    const size_t cap = 1024ull * 1024ull * 1024ull;
    size_t target = static_cast<size_t>(static_cast<double>(free_mem_bytes) * 0.20);
    target = std::min(cap, target);
    return std::max(kFp4WorkspaceMinBytes, target);
}

size_t clampFp4MatrixDim(size_t requested, double safety_ratio, size_t free_mem_bytes,
                         size_t workspace_bytes) {
    double usable_bytes = static_cast<double>(free_mem_bytes) * safety_ratio;
    if (usable_bytes <= static_cast<double>(workspace_bytes))
        return 16;

    // Device allocations: A, B, A_col, B_col (fp4) + output (fp32).
    double bytes_per_element = 4.0 + 4.0;
    double max_elems =
        (usable_bytes - static_cast<double>(workspace_bytes)) / bytes_per_element;

    size_t max_dim = static_cast<size_t>(std::sqrt(std::max(1.0, max_elems)));
    max_dim = (max_dim / 16) * 16;
    size_t requested_aligned = (std::max<size_t>(16, requested) / 16) * 16;

    if (max_dim < 16)
        return 16;
    return std::max<size_t>(16, std::min(requested_aligned, max_dim));
}
#endif

template <typename T>
ComputeResult runGemm(int device, Precision precision, const Options &opts,
                      size_t free_mem_bytes) {
    CUDA_CHECK(cudaSetDevice(device));
    size_t dim = clampMatrixDim(opts.matrix_dim, precision, 0.90, free_mem_bytes);
    size_t elems = dim * dim;
    size_t c_element_size = elementSize(precision);

    if (dim < opts.matrix_dim) {
        std::cout << "Device " << device << " adjusted matrix dim for "
                  << toString(precision) << " to " << dim << " (free memory bound)\n";
    }

    std::vector<T> hA(elems);
    std::vector<T> hB(elems);
    fillData(hA, precision);
    fillData(hB, precision);

    void *dA = nullptr;
    void *dB = nullptr;
    void *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, elems * elementSize(precision)));
    CUDA_CHECK(cudaMalloc(&dB, elems * elementSize(precision)));
    CUDA_CHECK(cudaMalloc(&dC, elems * c_element_size));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), elems * elementSize(precision), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), elems * elementSize(precision), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));

    cudaDataType_t dtype = dataType(precision);
    cublasComputeType_t ctype = computeType(precision, opts.tensor_cores);
    // Use FP32 scale type for all but FP64.
    cudaDataType_t scale_type = (precision == Precision::kFP64) ? CUDA_R_64F : CUDA_R_32F;
    cudaDataType_t output_type = dtype;

    cublasLtMatmulDesc_t op_desc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, ctype, scale_type));
    cublasOperation_t trans = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
    cublasLtMatrixLayout_t a_layout, b_layout, c_layout, d_layout;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_layout, dtype, dim, dim, dim));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_layout, dtype, dim, dim, dim));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_layout, output_type, dim, dim, dim));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&d_layout, output_type, dim, dim, dim));

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspace_limit = static_cast<size_t>(static_cast<double>(free_mem_bytes) * 0.10);
    const size_t workspace_cap = 64ull * 1024ull * 1024ull;
    workspace_limit = std::min(workspace_limit, workspace_cap);
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit)));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op_desc, a_layout, b_layout, c_layout, d_layout,
                                                pref, 1, &heuristic, &returned));
    if (returned == 0) {
        // Retry with zero workspace.
        size_t zero_ws = 0;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &zero_ws, sizeof(zero_ws)));
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op_desc, a_layout, b_layout, c_layout,
                                                    d_layout, pref, 1, &heuristic, &returned));
        if (returned == 0) {
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(d_layout);
            cublasLtMatrixLayoutDestroy(c_layout);
            cublasLtMatrixLayoutDestroy(b_layout);
            cublasLtMatrixLayoutDestroy(a_layout);
            cublasLtMatmulDescDestroy(op_desc);
            cublasLtDestroy(lt);
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
            throw std::runtime_error("No cuBLASLt heuristic available for GEMM");
        }
    }

    void *workspace_ptr = nullptr;
    if (heuristic.workspaceSize > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_ptr, heuristic.workspaceSize));
    }

    float alpha_f = 1.0f;
    float beta_f = 0.0f;
    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);
    double alpha_d = 1.0;
    double beta_d = 0.0;
    const void *alpha_ptr = nullptr;
    const void *beta_ptr = nullptr;
    if (scale_type == CUDA_R_64F) {
        alpha_ptr = static_cast<const void *>(&alpha_d);
        beta_ptr = static_cast<const void *>(&beta_d);
    } else {
        alpha_ptr = static_cast<const void *>(&alpha_f);
        beta_ptr = static_cast<const void *>(&beta_f);
    }

    // Warmup
    for (int i = 0; i < opts.warmup_iters; ++i) {
        CUBLAS_CHECK(cublasLtMatmul(lt, op_desc, alpha_ptr, dA, a_layout, dB, b_layout, beta_ptr,
                                    dC, c_layout, dC, d_layout, &heuristic.algo, workspace_ptr,
                                    heuristic.workspaceSize, nullptr));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double target_ms = opts.target_seconds > 0.0 ? opts.target_seconds * 1000.0 : 0.0;
    int total_iters = 0;
    float elapsed_ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    do {
        for (int i = 0; i < opts.compute_iters; ++i) {
            CUBLAS_CHECK(cublasLtMatmul(lt, op_desc, alpha_ptr, dA, a_layout, dB, b_layout,
                                        beta_ptr, dC, c_layout, dC, d_layout, &heuristic.algo,
                                        workspace_ptr, heuristic.workspaceSize, nullptr));
        }
        total_iters += opts.compute_iters;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    } while (target_ms > 0.0 && elapsed_ms < target_ms);

    double flops = 2.0 * static_cast<double>(dim) * static_cast<double>(dim) *
                   static_cast<double>(dim) * static_cast<double>(total_iters);
    double tflops = flops / (elapsed_ms / 1e3) / 1e12;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (workspace_ptr)
        cudaFree(workspace_ptr);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(d_layout);
    cublasLtMatrixLayoutDestroy(c_layout);
    cublasLtMatrixLayoutDestroy(b_layout);
    cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatmulDescDestroy(op_desc);
    cublasLtDestroy(lt);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    ComputeResult result;
    result.precision = precision;
    result.matrix_dim = dim;
    result.iterations = total_iters;
    result.tensor_cores = opts.tensor_cores && usesTensorCore(precision);
    result.throughput = tflops;
    result.throughput_unit = "TFLOP/s";
    result.elapsed_ms = elapsed_ms;
    return result;
}

#ifndef ENABLE_FP4_BENCH
ComputeResult runFp4GemmTe(int, const Options &, size_t) {
    throw std::runtime_error("FP4 support not enabled (rebuild with ENABLE_FP4_BENCH=1)");
}
#endif

ComputeResult runFp8GemmLt(int device, const Options &opts, size_t free_mem_bytes) {
    CUDA_CHECK(cudaSetDevice(device));
    size_t dim = clampMatrixDim(opts.matrix_dim, Precision::kFP8, 0.90, free_mem_bytes);
    size_t elems = dim * dim;
    if (dim < opts.matrix_dim) {
        std::cout << "Device " << device << " adjusted matrix dim for fp8 to " << dim
                  << " (free memory bound)\n";
    }

    std::vector<__nv_fp8_e4m3> hA(elems);
    std::vector<__nv_fp8_e4m3> hB(elems);
    fillData(hA, Precision::kFP8);
    fillData(hB, Precision::kFP8);

    __nv_fp8_e4m3 *dA = nullptr;
    __nv_fp8_e4m3 *dB = nullptr;
    float *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, elems * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&dB, elems * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&dC, elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), elems * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), elems * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t opDesc;
    // Peak FP8 Tensor Core with FP32 accumulate.
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
    cublasLtMatrixLayout_t aLayout, bLayout, cLayout, dLayout;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&aLayout, CUDA_R_8F_E4M3, dim, dim, dim));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&bLayout, CUDA_R_8F_E4M3, dim, dim, dim));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&cLayout, CUDA_R_32F, dim, dim, dim));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&dLayout, CUDA_R_32F, dim, dim, dim));

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspace_size = std::min<size_t>(64ull * 1024ull * 1024ull,
                                             static_cast<size_t>(free_mem_bytes * 0.10));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristics;
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, opDesc, aLayout, bLayout, cLayout, dLayout,
                                                pref, 1, &heuristics, &returned));
    if (returned == 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(dLayout);
        cublasLtMatrixLayoutDestroy(cLayout);
        cublasLtMatrixLayoutDestroy(bLayout);
        cublasLtMatrixLayoutDestroy(aLayout);
        cublasLtMatmulDescDestroy(opDesc);
        cublasLtDestroy(lt);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        throw std::runtime_error("No FP8 cublasLt heuristics found");
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    void *workspace_ptr = nullptr;
    if (heuristics.workspaceSize > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_ptr, heuristics.workspaceSize));
    }

    // Warmup
    for (int i = 0; i < opts.warmup_iters; ++i) {
        CUBLAS_CHECK(cublasLtMatmul(lt, opDesc, &alpha, dA, aLayout, dB, bLayout, &beta, dC,
                                    cLayout, dC, dLayout, &heuristics.algo, workspace_ptr,
                                    heuristics.workspaceSize, nullptr));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double target_ms = opts.target_seconds > 0.0 ? opts.target_seconds * 1000.0 : 0.0;
    int total_iters = 0;
    float elapsed_ms = 0.0f;

    CUDA_CHECK(cudaEventRecord(start));
    do {
        for (int i = 0; i < opts.compute_iters; ++i) {
            CUBLAS_CHECK(cublasLtMatmul(lt, opDesc, &alpha, dA, aLayout, dB, bLayout, &beta, dC,
                                        cLayout, dC, dLayout, &heuristics.algo, workspace_ptr,
                                        heuristics.workspaceSize, nullptr));
        }
        total_iters += opts.compute_iters;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    } while (target_ms > 0.0 && elapsed_ms < target_ms);

    double flops = 2.0 * static_cast<double>(dim) * static_cast<double>(dim) *
                   static_cast<double>(dim) * static_cast<double>(total_iters);
    double tflops = flops / (elapsed_ms / 1e3) / 1e12;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasLtMatmulPreferenceDestroy(pref);
    if (workspace_ptr)
        cudaFree(workspace_ptr);
    cublasLtMatrixLayoutDestroy(dLayout);
    cublasLtMatrixLayoutDestroy(cLayout);
    cublasLtMatrixLayoutDestroy(bLayout);
    cublasLtMatrixLayoutDestroy(aLayout);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(lt);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    ComputeResult result;
    result.precision = Precision::kFP8;
    result.matrix_dim = dim;
    result.iterations = total_iters;
    result.tensor_cores = true;
    result.throughput = tflops;
    result.throughput_unit = "TFLOP/s";
    result.elapsed_ms = elapsed_ms;
    return result;
}

#ifdef ENABLE_FP4_BENCH
ComputeResult runFp4GemmTe(int device, const Options &opts, size_t free_mem_bytes) {
#if !FP4_TYPE_SUPPORTED
    throw std::runtime_error("FP4 not supported by this CUDA toolkit (needs >= 12.8)");
#else
    using transformer_engine::DType;
    using transformer_engine::MatmulConfigWrapper;
    using transformer_engine::TensorWrapper;
    using transformer_engine::fp4e2m1;
    using transformer_engine::fp8e4m3;

    CUDA_CHECK(cudaSetDevice(device));
    size_t requested = std::max<size_t>(16, opts.matrix_dim);
    // FP4 kernels expect dimensions divisible by 16 and we also clamp to available memory.
    size_t workspace_bytes = fp4WorkspaceBytes(free_mem_bytes);
    size_t dim =
        clampFp4MatrixDim(requested, 0.85, free_mem_bytes, workspace_bytes);
    if (dim < requested) {
        std::cout << "Device " << device << " adjusted matrix dim for fp4 to " << dim
                  << " (free memory bound)\n";
    }

    auto round_up = [](size_t v, size_t m) { return ((v + m - 1) / m) * m; };

    size_t elems = dim * dim;
    std::vector<fp4e2m1> hA(elems);
    std::vector<fp4e2m1> hB(elems);
    std::vector<fp4e2m1> hA_col(elems);
    std::vector<fp4e2m1> hB_col(elems);
    for (size_t i = 0; i < elems; ++i) {
        float v = static_cast<float>((i % 17) - 8);
        hA[i] = fp4e2m1(v);
        hB[i] = fp4e2m1(v * 0.5f);
    }

    fp4e2m1 *dA = nullptr;
    fp4e2m1 *dB = nullptr;
    fp4e2m1 *dA_col = nullptr;
    fp4e2m1 *dB_col = nullptr;
    float *dOut = nullptr;
    fp8e4m3 *scale_inv_row = nullptr;
    fp8e4m3 *scale_inv_col = nullptr;
    void *workspace_ptr = nullptr;
    const size_t workspace_bytes_final = workspace_bytes;
    CUDA_CHECK(cudaMalloc(&dA, elems * sizeof(fp4e2m1)));
    CUDA_CHECK(cudaMalloc(&dB, elems * sizeof(fp4e2m1)));
    CUDA_CHECK(cudaMalloc(&dA_col, elems * sizeof(fp4e2m1)));
    CUDA_CHECK(cudaMalloc(&dB_col, elems * sizeof(fp4e2m1)));
    CUDA_CHECK(cudaMalloc(&dOut, elems * sizeof(float)));
    size_t scale_outer_row = round_up(dim, 128);
    size_t scale_inner_row = round_up((dim + 15) / 16, 4);
    size_t scale_outer_col = round_up(dim, 128);
    size_t scale_inner_col = round_up((dim + 15) / 16, 4);
    size_t scale_row_elems = scale_outer_row * scale_inner_row;
    size_t scale_col_elems = scale_outer_col * scale_inner_col;
    std::vector<fp8e4m3> h_scale_row(scale_row_elems, fp8e4m3(1.0f));
    std::vector<fp8e4m3> h_scale_col(scale_col_elems, fp8e4m3(1.0f));
    CUDA_CHECK(cudaMalloc(&scale_inv_row, scale_row_elems * sizeof(fp8e4m3)));
    CUDA_CHECK(cudaMalloc(&scale_inv_col, scale_col_elems * sizeof(fp8e4m3)));
    CUDA_CHECK(cudaMemcpy(scale_inv_row, h_scale_row.data(),
                          scale_row_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scale_inv_col, h_scale_col.data(),
                          scale_col_elems * sizeof(fp8e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&workspace_ptr, workspace_bytes_final));
    bool cleaned = false;
    auto cleanup = [&]() {
        if (cleaned)
            return;
        cleaned = true;
        cudaFree(dA_col);
        cudaFree(dB_col);
        cudaFree(workspace_ptr);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dOut);
        cudaFree(scale_inv_row);
        cudaFree(scale_inv_col);
    };

    // Prepare columnwise (transposed) copies for NVFP4 requirements.
    for (size_t row = 0; row < dim; ++row) {
        for (size_t col = 0; col < dim; ++col) {
            hA_col[col * dim + row] = hA[row * dim + col];
            hB_col[col * dim + row] = hB[row * dim + col];
        }
    }

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), elems * sizeof(fp4e2m1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), elems * sizeof(fp4e2m1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_col, hA_col.data(), elems * sizeof(fp4e2m1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_col, hB_col.data(), elems * sizeof(fp4e2m1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dOut, 0, elems * sizeof(float)));

    const size_t mat_dims[] = {dim, dim};
    const size_t scale_dims[] = {scale_outer_row, scale_inner_row};
    const size_t scale_col_dims[] = {scale_outer_col, scale_inner_col};
    auto mat_shape = nvte_make_shape(mat_dims, 2);
    auto scale_shape = nvte_make_shape(scale_dims, 2);
    auto scale_col_shape = nvte_make_shape(scale_col_dims, 2);

    TensorWrapper A_wrapper(NVTE_NVFP4_1D_SCALING);
    TensorWrapper B_wrapper(NVTE_NVFP4_1D_SCALING);
    A_wrapper.set_rowwise_data(dA, DType::kFloat4E2M1, mat_shape)
        .set_rowwise_scale_inv(scale_inv_row, DType::kFloat8E4M3, scale_shape)
        .set_columnwise_data(dA_col, DType::kFloat4E2M1, mat_shape)
        .set_columnwise_scale_inv(scale_inv_col, DType::kFloat8E4M3, scale_col_shape);
    B_wrapper.set_rowwise_data(dB, DType::kFloat4E2M1, mat_shape)
        .set_rowwise_scale_inv(scale_inv_row, DType::kFloat8E4M3, scale_shape)
        .set_columnwise_data(dB_col, DType::kFloat4E2M1, mat_shape)
        .set_columnwise_scale_inv(scale_inv_col, DType::kFloat8E4M3, scale_col_shape);
    // TE currently requires C and D to alias for NVFP4 path.
    TensorWrapper CD_wrapper(dOut, mat_shape, DType::kFloat32, nullptr, nullptr, nullptr,
                             scale_shape, NVTE_DELAYED_TENSOR_SCALING);
    TensorWrapper workspace(workspace_ptr, std::vector<size_t>{workspace_bytes_final / sizeof(float)},
                            DType::kFloat32);
    MatmulConfigWrapper config;

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaEvent_t start, stop;
    bool events_created = false;
    try {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        events_created = true;

        // Warmup
        for (int i = 0; i < opts.warmup_iters; ++i) {
            nvte_cublas_gemm_v2(/*transa*/ 0, /*transb*/ 0, &alpha, A_wrapper.data(),
                                B_wrapper.data(), &beta, CD_wrapper.data(), CD_wrapper.data(),
                                workspace.data(), config, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        double target_ms = opts.target_seconds > 0.0 ? opts.target_seconds * 1000.0 : 0.0;
        int total_iters = 0;
        float elapsed_ms = 0.0f;

        CUDA_CHECK(cudaEventRecord(start));
        do {
            for (int i = 0; i < opts.compute_iters; ++i) {
                nvte_cublas_gemm_v2(/*transa*/ 0, /*transb*/ 0, &alpha, A_wrapper.data(),
                                    B_wrapper.data(), &beta, CD_wrapper.data(), CD_wrapper.data(),
                                    workspace.data(), config, 0);
            }
            total_iters += opts.compute_iters;
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        } while (target_ms > 0.0 && elapsed_ms < target_ms);

        double flops = 2.0 * static_cast<double>(dim) * static_cast<double>(dim) *
                       static_cast<double>(dim) * static_cast<double>(total_iters);
        double tflops = flops / (elapsed_ms / 1e3) / 1e12;

        if (events_created) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        cleanup();

        ComputeResult result;
        result.precision = Precision::kFP4;
        result.matrix_dim = dim;
        result.iterations = total_iters;
        result.tensor_cores = true;
        result.throughput = tflops; // fp4 uses TFLOP/s units
        result.throughput_unit = "TFLOP/s";
        result.elapsed_ms = elapsed_ms;
        return result;
    } catch (...) {
        if (events_created) {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        cleanup();
        cudaGetLastError(); // clear sticky error state
        throw;
    }

#endif
}
#endif // ENABLE_FP4_BENCH

__global__ void streamingKernel(const float *src, float *dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        float v = src[i];
        dst[i] = v + 1.0f;
    }
}

MemoryResult runMemory(int device, const Options &opts, size_t free_mem_bytes) {
    CUDA_CHECK(cudaSetDevice(device));
    size_t target_bytes = opts.mem_bytes > 0
                              ? static_cast<size_t>(opts.mem_bytes)
                              : static_cast<size_t>(free_mem_bytes * (opts.mem_percent / 100.0));
    // Leave headroom for fragmentation and resident allocations; then retry on failure.
    const double headroom = 0.80; // start with 80% of reported free memory
    size_t usable_bytes = static_cast<size_t>(static_cast<double>(free_mem_bytes) * headroom);
    if (target_bytes > usable_bytes)
        target_bytes = usable_bytes;
    const size_t min_bytes = 64ull * 1024ull * 1024ull;
    float *src = nullptr;
    float *dst = nullptr;
    size_t aligned_bytes = target_bytes - (target_bytes % sizeof(float));
    cudaError_t alloc_err = cudaSuccess;
    while (aligned_bytes >= min_bytes) {
        alloc_err = cudaMalloc(&src, aligned_bytes);
        if (alloc_err == cudaSuccess) {
            alloc_err = cudaMalloc(&dst, aligned_bytes);
            if (alloc_err == cudaSuccess)
                break;
            cudaFree(src);
        }
        aligned_bytes = static_cast<size_t>(static_cast<double>(aligned_bytes) * 0.8);
        aligned_bytes -= aligned_bytes % sizeof(float);
    }
    if (alloc_err != cudaSuccess) {
        throw std::runtime_error("Memory test allocation failed after retries: "
                                 + std::to_string(target_bytes / (1024 * 1024)) + " MB target");
    }

    std::cout << "Device " << device << " memory test buffer: "
              << aligned_bytes / (1024 * 1024) << " MB\n";

    CUDA_CHECK(cudaMemset(dst, 0, aligned_bytes));
    CUDA_CHECK(cudaMemset(dst, 0, aligned_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double target_ms = opts.target_seconds > 0.0 ? opts.target_seconds * 1000.0 : 0.0;

    // D2D memcpy bandwidth (pattern lifted from cuda-samples bandwidthTest).
    CUDA_CHECK(cudaEventRecord(start));
    int d2d_iters = 0;
    float d2d_ms = 0.0f;
    do {
        for (int i = 0; i < opts.mem_iters; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, aligned_bytes, cudaMemcpyDeviceToDevice));
        }
        d2d_iters += opts.mem_iters;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&d2d_ms, start, stop));
    } while (target_ms > 0.0 && d2d_ms < target_ms);


    // Streaming kernel bandwidth (global load/store bound).
    size_t elem_count = aligned_bytes / sizeof(float);
    int threads = 256;
    int blocks = std::min<int>(static_cast<int>((elem_count + threads - 1) / threads), 4096);
    CUDA_CHECK(cudaEventRecord(start));
    int stream_iters = 0;
    float stream_ms = 0.0f;
    do {
        for (int i = 0; i < opts.mem_iters; ++i) {
            streamingKernel<<<blocks, threads>>>(src, dst, elem_count);
        }
        stream_iters += opts.mem_iters;
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&stream_ms, start, stop));
    } while (target_ms > 0.0 && stream_ms < target_ms);
    CUDA_CHECK(cudaGetLastError());

    double d2d_bytes = static_cast<double>(aligned_bytes) * 2.0 * d2d_iters;
    double stream_bytes = static_cast<double>(aligned_bytes) * 2.0 * stream_iters;

    MemoryResult res;
    res.bytes = aligned_bytes;
    res.d2d_iterations = d2d_iters;
    res.streaming_iterations = stream_iters;
    res.d2d_bandwidth_gbs = d2d_bytes / (d2d_ms / 1e3) / 1e9;
    res.streaming_bandwidth_gbs = stream_bytes / (stream_ms / 1e3) / 1e9;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(src);
    cudaFree(dst);
    return res;
}

DeviceReport runForDevice(int device, const Options &opts) {
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    size_t free_mem = 0;
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    DeviceReport rep;
    rep.device_index = device;
    rep.name = prop.name;
    rep.sm_major = prop.major;
    rep.sm_minor = prop.minor;
    rep.total_mem = total_mem;
    rep.free_mem = free_mem;

    if (opts.run_compute) {
        for (Precision p : opts.precisions) {
            try {
                switch (p) {
                case Precision::kFP32:
                    rep.compute.push_back(runGemm<float>(device, p, opts, free_mem));
                    break;
                case Precision::kFP16:
                    rep.compute.push_back(runGemm<__half>(device, p, opts, free_mem));
                    break;
                case Precision::kBF16:
                    rep.compute.push_back(runGemm<__nv_bfloat16>(device, p, opts, free_mem));
                    break;
                case Precision::kFP8:
                    rep.compute.push_back(runFp8GemmLt(device, opts, free_mem));
                    break;
                case Precision::kFP4:
#ifdef ENABLE_FP4_BENCH
                    try {
                        rep.compute.push_back(runFp4GemmTe(device, opts, free_mem));
                    } catch (const std::exception &e) {
                        cudaGetLastError(); // clear sticky errors before deciding on fallback
                        bool illegal = std::string(e.what()).find("illegal") != std::string::npos;
                        if (illegal && opts.matrix_dim > 4096) {
                            Options smaller = opts;
                            smaller.matrix_dim = std::max<size_t>(4096, (opts.matrix_dim / 2 / 16) * 16);
                            std::cerr << "Retrying fp4 on device " << device << " with matrix "
                                      << smaller.matrix_dim << " after error: " << e.what()
                                      << "\n";
                            rep.compute.push_back(runFp4GemmTe(device, smaller, free_mem));
                        } else {
                            throw;
                        }
                    }
#else
                    throw std::runtime_error("FP4 requested but this build lacks ENABLE_FP4_BENCH=1");
#endif
                    break;
                case Precision::kFP64:
                    rep.compute.push_back(runGemm<double>(device, p, opts, free_mem));
                    break;
                }
            } catch (const std::exception &e) {
                std::cerr << "Skipping " << toString(p) << " on device " << device
                          << ": " << e.what() << "\n";
            }
        }
    }

    if (opts.run_memory) {
        try {
            rep.memory = runMemory(device, opts, free_mem);
        } catch (const std::exception &e) {
            std::cerr << "Skipping memory test on device " << device << ": " << e.what()
                      << "\n";
        }
    }

    return rep;
}

void printReport(const std::vector<DeviceReport> &reports) {
    for (const auto &rep : reports) {
        std::cout << "\nGPU " << rep.device_index << " (" << rep.name << ", sm "
                  << rep.sm_major << rep.sm_minor << ")\n";
        if (!rep.compute.empty()) {
            std::cout << "  Compute:\n";
            for (const auto &c : rep.compute) {
                std::cout << "    " << toString(c.precision) << " ("
                          << c.matrix_dim << "): " << std::fixed << std::setprecision(1)
                          << c.throughput << " " << c.throughput_unit
                          << " in " << std::setprecision(2) << c.elapsed_ms << " ms"
                          << (c.tensor_cores ? " (tensor cores)" : "") << "\n";
            }
        }
        if (rep.memory) {
            std::cout << "  Memory bandwidth (" << rep.memory->bytes / (1024 * 1024)
                      << " MB buffer):\n";
            std::cout << "    D2D memcpy : " << std::fixed << std::setprecision(1)
                      << rep.memory->d2d_bandwidth_gbs << " GB/s ("
                      << rep.memory->d2d_iterations << " iters)\n";
            std::cout << "    Streaming  : " << std::fixed << std::setprecision(1)
                      << rep.memory->streaming_bandwidth_gbs << " GB/s ("
                      << rep.memory->streaming_iterations << " iters)\n";
        }
    }
}

std::string jsonEscape(const std::string &in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == '"' || c == '\\')
            out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

void writeJson(const std::string &path, const std::vector<DeviceReport> &reports,
               const Options &opts) {
    std::ofstream out(path);
    if (!out.good()) {
        std::cerr << "Could not open " << path << " for writing\n";
        return;
    }

    out << "{\n";
    out << "  \"matrix_dim\": " << opts.matrix_dim << ",\n";
    out << "  \"compute_iterations\": " << opts.compute_iters << ",\n";
    out << "  \"mem_iterations\": " << opts.mem_iters << ",\n";
    out << "  \"target_seconds\": " << opts.target_seconds << ",\n";
    out << "  \"tensor_cores\": " << (opts.tensor_cores ? "true" : "false") << ",\n";
    out << "  \"parallel\": " << (opts.parallel ? "true" : "false") << ",\n";
    out << "  \"devices\": [\n";
    for (size_t i = 0; i < reports.size(); ++i) {
        const auto &rep = reports[i];
        out << "    {\n";
        out << "      \"cuda_index\": " << rep.device_index << ",\n";
        out << "      \"name\": \"" << jsonEscape(rep.name) << "\",\n";
        out << "      \"sm\": \"" << rep.sm_major << "." << rep.sm_minor << "\",\n";
        out << "      \"total_memory_bytes\": " << rep.total_mem << ",\n";
        out << "      \"free_memory_bytes\": " << rep.free_mem << ",\n";
        out << "      \"compute\": [\n";
        for (size_t c = 0; c < rep.compute.size(); ++c) {
            const auto &cr = rep.compute[c];
            out << "        {\n";
            out << "          \"precision\": \"" << toString(cr.precision) << "\",\n";
            out << "          \"matrix_dim\": " << cr.matrix_dim << ",\n";
            out << "          \"iterations\": " << cr.iterations << ",\n";
            out << "          \"tensor_cores\": "
                << (cr.tensor_cores ? "true" : "false") << ",\n";
            out << "          \"throughput\": " << cr.throughput << ",\n";
            out << "          \"throughput_unit\": \"" << cr.throughput_unit << "\",\n";
            out << "          \"elapsed_ms\": " << cr.elapsed_ms << "\n";
            out << "        }" << (c + 1 == rep.compute.size() ? "\n" : ",\n");
        }
        out << "      ],\n";
        out << "      \"memory\": ";
        if (rep.memory) {
            out << "{\n";
            out << "        \"bytes\": " << rep.memory->bytes << ",\n";
            out << "        \"d2d_iterations\": " << rep.memory->d2d_iterations << ",\n";
            out << "        \"streaming_iterations\": " << rep.memory->streaming_iterations
                << ",\n";
            out << "        \"d2d_bandwidth_gbs\": " << rep.memory->d2d_bandwidth_gbs
                << ",\n";
            out << "        \"streaming_bandwidth_gbs\": "
                << rep.memory->streaming_bandwidth_gbs << "\n";
            out << "      }\n";
        } else {
            out << "null\n";
        }
        out << "    }" << (i + 1 == reports.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

Options parseArgs(int argc, char **argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printHelp();
            std::exit(0);
        } else if (arg == "--list" || arg == "-l") {
            opts.list = true;
        } else if (arg == "--devices") {
            if (i + 1 >= argc)
                throw std::runtime_error("--devices needs a value");
            opts.devices = parseDeviceList(argv[++i]);
        } else if (arg == "--precisions") {
            if (i + 1 >= argc)
                throw std::runtime_error("--precisions needs a value");
            opts.precisions.clear();
            for (const auto &token : split(argv[++i], ',')) {
                opts.precisions.push_back(parsePrecision(token));
            }
        } else if (arg == "--matrix") {
            if (i + 1 >= argc)
                throw std::runtime_error("--matrix needs a value");
            opts.matrix_dim = std::stoul(argv[++i]);
        } else if (arg == "--compute-iters") {
            if (i + 1 >= argc)
                throw std::runtime_error("--compute-iters needs a value");
            opts.compute_iters = std::stoi(argv[++i]);
        } else if (arg == "--mem-iters") {
            if (i + 1 >= argc)
                throw std::runtime_error("--mem-iters needs a value");
            opts.mem_iters = std::stoi(argv[++i]);
        } else if (arg == "--mem") {
            if (i + 1 >= argc)
                throw std::runtime_error("--mem needs a value");
            parseMemoryArg(argv[++i], opts);
        } else if (arg == "--seconds" || arg == "--duration") {
            if (i + 1 >= argc)
                throw std::runtime_error("--seconds needs a value");
            opts.target_seconds = std::stod(argv[++i]);
        } else if (arg == "--parallel") {
            opts.parallel = true;
        } else if (arg == "--no-compute") {
            opts.run_compute = false;
        } else if (arg == "--no-memory") {
            opts.run_memory = false;
        } else if (arg == "--no-tensor-cores") {
            opts.tensor_cores = false;
        } else if (arg == "--report") {
            if (i + 1 >= argc)
                throw std::runtime_error("--report needs a path");
            opts.report_path = argv[++i];
        } else {
            std::ostringstream oss;
            oss << "Unknown argument: " << arg;
            throw std::runtime_error(oss.str());
        }
    }
    if (opts.compute_iters <= 0)
        throw std::runtime_error("--compute-iters must be > 0");
    if (opts.mem_iters <= 0)
        throw std::runtime_error("--mem-iters must be > 0");
    if (opts.target_seconds < 0.0)
        throw std::runtime_error("--seconds/--duration must be >= 0");
    return opts;
}

void listDevices() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        size_t free_mem = 0;
        size_t total_mem = 0;
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        std::cout << "GPU " << i << ": " << prop.name << " (sm " << prop.major << prop.minor
                  << "), memory " << total_mem / (1024 * 1024) << " MB free "
                  << free_mem / (1024 * 1024) << " MB\n";
    }
}

std::vector<int> resolveDevices(const Options &opts) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (opts.devices.empty())
        return [] (int c) {
            std::vector<int> ids;
            ids.reserve(c);
            for (int i = 0; i < c; ++i)
                ids.push_back(i);
            return ids;
        }(count);

    std::vector<int> filtered;
    for (int id : opts.devices) {
        if (id >= 0 && id < count)
            filtered.push_back(id);
    }
    return filtered;
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options opts = parseArgs(argc, argv);
        if (opts.list) {
            listDevices();
            return 0;
        }

        auto devices = resolveDevices(opts);
        if (devices.empty()) {
            std::cerr << "No CUDA devices selected.\n";
            return 1;
        }

        std::vector<DeviceReport> reports_storage(devices.size());
        std::vector<char> success(devices.size(), 0);
        std::vector<std::string> errors;
        std::mutex error_mutex;

        if (opts.parallel && devices.size() > 1) {
            std::vector<std::thread> workers;
            workers.reserve(devices.size());
            for (size_t idx = 0; idx < devices.size(); ++idx) {
                workers.emplace_back([&, idx]() {
                    try {
                        reports_storage[idx] = runForDevice(devices[idx], opts);
                        success[idx] = 1;
                    } catch (const std::exception &e) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        std::ostringstream oss;
                        oss << "device " << devices[idx] << ": " << e.what();
                        errors.push_back(oss.str());
                    }
                });
            }
            for (auto &t : workers)
                t.join();
        } else {
            for (size_t idx = 0; idx < devices.size(); ++idx) {
                reports_storage[idx] = runForDevice(devices[idx], opts);
                success[idx] = 1;
            }
        }

        if (!errors.empty()) {
            for (const auto &msg : errors) {
                std::cerr << "Failed device run: " << msg << "\n";
            }
        }

        std::vector<DeviceReport> reports;
        reports.reserve(devices.size());
        for (size_t idx = 0; idx < devices.size(); ++idx) {
            if (success[idx])
                reports.push_back(std::move(reports_storage[idx]));
        }

        if (reports.empty()) {
            std::cerr << "No successful device runs.\n";
            return 1;
        }

        printReport(reports);
        if (!opts.report_path.empty())
            writeJson(opts.report_path, reports, opts);
    } catch (const std::exception &e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
