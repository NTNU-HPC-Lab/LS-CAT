#include "includes.h"
__global__ void sum4(float4 *A, float4 *B, float4 *C, const int N)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < N)
{ C[i].x = A[i].x + B[i].x;C[i].y = A[i].y + B[i].y;C[i].z = A[i].z + B[i].z;C[i].w = A[i].w + B[i].w;}
}