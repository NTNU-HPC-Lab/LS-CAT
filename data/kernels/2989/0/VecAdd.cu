#include "includes.h"



// Kernel definition

__global__ void VecAdd(float* A, float* B, float* C,int nums)
{
int i = blockDim.x * blockIdx.x + threadIdx.x;
//C[i] = A[i] + B[i];
if(i<nums){
//C[i]=A[i]+B[i];
C[i]=static_cast<float>(i);
}
}