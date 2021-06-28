#include "includes.h"
__global__ void cudaDTargetBiasPropagate_kernel( unsigned int size, const double bias, const double* inputs, const double* diffInputs, double* outputs)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
outputs[i] = inputs[i];

if (diffInputs[i] > 0.0 && inputs[i] > -bias)
outputs[i] += bias;
}
}