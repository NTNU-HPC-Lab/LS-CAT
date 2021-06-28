#include "includes.h"
#ifdef __cplusplus
extern "C" {
#endif






#ifdef __cplusplus
}
#endif
__global__ void vec_add(float *A, float *B, float* C, int size)
{
int index = blockIdx.x*blockDim.x + threadIdx.x;

if(index<size)
C[index] = A[index] + B[index];
}