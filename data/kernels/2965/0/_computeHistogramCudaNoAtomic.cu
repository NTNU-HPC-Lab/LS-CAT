#include "includes.h"



using namespace std;

using HistType = uint32_t;

enum class Mode {
CPU,
OMP,
OMP_NOATOMIC,
CUDA,
CUDA_NOATOMIC,
CUDA_SHARED,
};

enum class AtomicTypeCuda {
NONE,
STANDARD,
SHARED,
};

__global__ void _computeHistogramCudaNoAtomic(const uint8_t *__restrict__ bytes, size_t length, HistType *__restrict__ histogram) {
size_t stride = blockDim.x * gridDim.x;

for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
i += stride) {
histogram[bytes[i]]++;
}
}