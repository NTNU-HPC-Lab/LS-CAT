#include "includes.h"

#define SIZE_thread 1024



__global__ void VectorAdd(int *A, int *B, int *C,int n)
{
int i = threadIdx.x + blockIdx.x*blockDim.x;
if(i<n)
C[i]=A[i]+B[i];
}