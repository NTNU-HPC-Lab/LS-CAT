#include "includes.h"
__global__ void kSigmoid(const int nThreads, float const *input, float *output){
/*  Computes the value of the sigmoid function f(x) = 1/(1 + e^-x).
Inputs:
input: array
output: array, the results of the computation are to be stored here
*/

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
i < nThreads;
i += blockDim.x * gridDim.x)
{
output[i] = 1.0 / (1.0 + std::exp(-input[i]));
}
}