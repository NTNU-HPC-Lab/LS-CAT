#include "includes.h"
__global__ void sumGrad(float* output, float* input1, float* input2, float* input3, float* input4, const int numElem)
{
size_t pos = blockDim.x * blockIdx.x + threadIdx.x;
size_t size = blockDim.x * gridDim.x;

for(int i = numElem * pos / size; i < numElem * (pos+1) / size; i++){
output[i] = input1[i] + input2[i] + input3[i] + input4[i];
}
}