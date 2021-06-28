#include "includes.h"
__global__ void vectorAddKernel(float* inputA, float* inputB, float* output, int length){

//compute element index
int idx = blockIdx.x * blockDim.x + threadIdx.x;

//add an vector element
if(idx < length) output[idx] = inputA[idx] + inputB[idx];

}