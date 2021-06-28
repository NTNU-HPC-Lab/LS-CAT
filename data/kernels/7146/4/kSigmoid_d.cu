#include "includes.h"
__global__ void kSigmoid_d(const int nThreads, float const *input, float *output) {
/*  Computes the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
where f(x) is sigmoid function.
Inputs:
input: array
output: array, the results of the computation are to be stored here:
x(1 - x) for every element of the input matrix m1.
*/

for (int i = blockIdx.x * blockDim.x + threadIdx.x;
i < nThreads;
i += blockDim.x * gridDim.x)
{
output[i] = input[i] * (1 - input[i]);
}
}