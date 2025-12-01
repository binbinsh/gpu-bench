ifneq ("$(wildcard /usr/bin/nvcc)", "")
CUDAPATH ?= /usr
else ifneq ("$(wildcard /usr/local/cuda/bin/nvcc)", "")
CUDAPATH ?= /usr/local/cuda
endif

IS_JETSON   ?= $(shell if grep -Fwq "Jetson" /proc/device-tree/model 2>/dev/null; then echo true; else echo false; fi)
NVCC        :=  ${CUDAPATH}/bin/nvcc
CCPATH      ?=
NVCC_CCBIN  := $(if ${CCPATH},-ccbin ${CCPATH})

# Detect CUDA version (major.minor)
CUDA_VERSION_DETECTED := $(shell ${NVCC} --version | grep "release" | sed -E 's/.*release ([0-9]+\.[0-9]+).*/\1/')
CUDA_VERSION_MAJOR := $(shell echo ${CUDA_VERSION_DETECTED} | cut -d. -f1)
CUDA_VERSION_MINOR := $(shell echo ${CUDA_VERSION_DETECTED} | cut -d. -f2)

# Prefer GCC 12 if present (nvcc supported)
ifeq ($(CCPATH),)
ifneq ("$(wildcard /usr/bin/g++-12)","")
CCPATH := /usr/bin/g++-12
NVCC_CCBIN := -ccbin ${CCPATH}
endif
endif
ENABLE_FP4 ?= 0
TE_ROOT ?= transformer-engine
TE_INCLUDEDIR ?= ${TE_ROOT}/transformer_engine/common/include
TE_SRCDIR ?= ${TE_ROOT}
TE_LIBDIR ?= ${TE_ROOT}/build
LDFLAGS_FP4 :=
GPU_BENCH_DEPS :=

override CFLAGS   ?=
override CFLAGS   += -O3
override CFLAGS   += -Wno-unused-result
override CFLAGS   += -I${CUDAPATH}/include
override CFLAGS   += -std=c++11
override CFLAGS   += -DIS_JETSON=${IS_JETSON}

override LDFLAGS  ?=
override LDFLAGS  += -lcuda
override LDFLAGS  += -L${CUDAPATH}/lib64
override LDFLAGS  += -L${CUDAPATH}/lib64/stubs
override LDFLAGS  += -L${CUDAPATH}/lib
override LDFLAGS  += -L${CUDAPATH}/lib/stubs
override LDFLAGS  += -Xlinker -rpath -Xlinker ${CUDAPATH}/lib64
override LDFLAGS  += -Xlinker -rpath -Xlinker ${CUDAPATH}/lib
override LDFLAGS  += -lcublas
override LDFLAGS  += -lcublasLt
override LDFLAGS  += -lcudart

COMPUTE      ?= 89
# Allow comma-separated architectures directly via COMPUTE, e.g. COMPUTE=89,120
COMMA        := ,
ARCH_LIST    := $(strip $(subst $(COMMA), ,$(subst .,,$(COMPUTE))))
CUDA_VERSION ?= 11.8.0
IMAGE_DISTRO ?= ubi8

override NVCCFLAGS ?=
override NVCCFLAGS += -I${CUDAPATH}/include
override NVCCFLAGS += $(foreach arch,${ARCH_LIST},-gencode arch=compute_${arch},code=sm_${arch})
override NVCCFLAGS += -std=c++17
ALLOW_UNSUPPORTED_COMPILER ?= 1
ifeq (${ALLOW_UNSUPPORTED_COMPILER},1)
override NVCCFLAGS += --allow-unsupported-compiler
endif
ifeq (${ENABLE_FP4},1)
override NVCCFLAGS += -DENABLE_FP4_BENCH -I${TE_INCLUDEDIR} -I${TE_SRCDIR}
LDFLAGS_FP4 += -L${TE_LIBDIR} -Xlinker -rpath -Xlinker ${TE_LIBDIR} -ltransformer_engine -lnvrtc
GPU_BENCH_DEPS += ${TE_LIBDIR}/libtransformer_engine.so
endif

IMAGE_NAME ?= gpu-bench

.PHONY: all

all: gpu_bench

gpu_bench: benchmarks/quantized_benchmark.cu ${GPU_BENCH_DEPS}
	${NVCC} ${NVCCFLAGS} ${NVCC_CCBIN} -O3 -o $@ $< ${LDFLAGS_FP4} ${LDFLAGS}

%.o: %.cpp
	g++ ${CFLAGS} -c $<

.PHONY: clean
clean:
	$(RM) *.ptx *.o gpu_bench

image:
	docker build --build-arg CUDA_VERSION=${CUDA_VERSION} --build-arg IMAGE_DISTRO=${IMAGE_DISTRO} -t ${IMAGE_NAME} .
