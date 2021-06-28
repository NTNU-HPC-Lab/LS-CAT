#include "includes.h"
__global__ void VecAdd(double* A,double* B,double* C)
{
// extern __shared__ float sdata[];
int i=threadIdx.x;
C[i]=A[i]+B[i];

}