#include "includes.h"
__global__ void postprocess_matrix(float* matrix, long* long_indices, int* indices, unsigned int N_POINTS, unsigned int K)
{
register int TID = threadIdx.x + blockIdx.x * blockDim.x;
if (TID >= N_POINTS*K) return;

// Set pij to 0 for each of the broken values - Note: this should be handled in the ComputePijKernel now
// if (matrix[TID] == 1.0f) matrix[TID] = 0.0f;
indices[TID] = (int) long_indices[TID];
return;
}