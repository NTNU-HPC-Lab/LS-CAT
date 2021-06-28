#include "includes.h"
__global__ void cudaSTargetBiasPropagate_kernel( unsigned int size, const float bias, const float* inputs, const float* diffInputs, float* outputs)
{
const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int stride = blockDim.x * gridDim.x;

for (unsigned int i = index; i < size; i += stride) {
outputs[i] = inputs[i];

if (diffInputs[i] > 0.0f && inputs[i] > -bias)
outputs[i] += bias;
}
}