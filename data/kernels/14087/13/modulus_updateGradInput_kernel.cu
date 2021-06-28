#include "includes.h"
__global__ void modulus_updateGradInput_kernel(float* input, float* output, float* gradInput, float* gradOutput, int n) {
const int i = threadIdx.x + blockIdx.x*blockDim.x;
if (i >= n)
return;
const float eps = 0.0001;
const float c = gradOutput[i]/max(output[i],eps);
gradInput[2*i] = input[2*i]*c;
gradInput[2*i+1] = input[2*i+1]*c;
}