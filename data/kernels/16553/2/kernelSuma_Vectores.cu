#include "includes.h"
__global__ void kernelSuma_Vectores(float* array_A, float* array_B, int _size){
int idx= blockIdx.x*blockDim.x+threadIdx.x;
if(idx<_size){
array_A[idx] = array_A[idx] + array_B[idx];
}
}