#include "includes.h"
__device__ int f () { return 21; }
__global__ void vDisp(const float *A, const float *B, int ds)
{
int idx = blockIdx.x * block_size + threadIdx.x; // create typical 1D thread index from built-in variables
printf("idx = %d, ds = %d\n", idx, ds);
if (idx < ds)
printf("Device: [%d], \t%f\t%f \n", idx, A[idx], B[idx]);         // do the vector (element) add here
}