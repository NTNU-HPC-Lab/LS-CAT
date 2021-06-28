#include "includes.h"
// Header files

// Routine to check error
__global__ void sum(int *A , int *B, int *C, long long N)
{
long long idx = blockIdx.x * blockDim.x + threadIdx.x ;
if(idx  < N)
{
C[idx] = A[idx] + B[idx] ;
}
}