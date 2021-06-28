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

__global__ void _computeHistogramCudaSharedAtomic(const uint8_t *__restrict__ bytes, size_t length, HistType *__restrict__ histogram) {
__shared__ HistType temp[256];
temp[threadIdx.x] = 0;
__syncthreads(); // Zero this block's temporary array

size_t stride = blockDim.x * gridDim.x;

for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
i += stride) {
atomicAdd(&(temp[bytes[i]]), 1u);
// Make a histogram for a fraction of the bytes
}
__syncthreads();

// Now add up the histograms
atomicAdd(&(histogram[threadIdx.x]), temp[threadIdx.x]);

// Lesson: Don't let too many threads touch the same memory addresses at once
}