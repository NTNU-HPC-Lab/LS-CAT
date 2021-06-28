#include "includes.h"
/*
* Compile: nvcc -o saxby saxby.cu
* Run: ./saxby
*/
__global__ void daxbyAdd(const float *A, const float *B, float *C, float x,int numElements){
int i = blockDim.x * blockIdx.x + threadIdx.x;
if(i < numElements){
C[i] = A[i]* x + B[i];
}
}