#include "includes.h"

extern "C" {
}

/**
* CUDA Kernel Device code
*
* Computes the vector addition of A and B into C. The 3 vectors have the same
* number of elements numElements.
*/

typedef struct {
float *hA, *hB, *hC;
float *dA, *dB, *dC;
int element_count;
size_t vector_bytes;
int v_threadsPerBlock;
int v_blocksPerGrid;
cudaStream_t stream;
} ThreadContext;

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
int i = blockDim.x * blockIdx.x + threadIdx.x;

if (i < numElements)
{
C[i] = A[i] + B[i];
}
}