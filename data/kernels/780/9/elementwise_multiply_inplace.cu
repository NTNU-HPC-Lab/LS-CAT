#include "includes.h"
__global__ void elementwise_multiply_inplace(const cuDoubleComplex* A, cuDoubleComplex* B, const int size)
{
int tid = blockIdx.x * blockDim.x + threadIdx.x;
if(tid < size) {
B[tid] = cuCmul(A[tid], B[tid]);
}
}