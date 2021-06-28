#include "includes.h"
__global__ void softmax_loss_kernel(float *reduced_loss, float *predict, float *target, float *workspace, int batch_size, int num_outputs)
{
int batch_idx = blockDim.x * blockIdx.x + threadIdx.x;

extern __shared__ float s_data[];
float loss = 0.f;

// each thread calculate entropy for each data and accumulate to shared memory
for (int c = 0; c < num_outputs; c++)
loss += target[batch_idx * num_outputs + c] * logf(predict[batch_idx * num_outputs + c]);
workspace[batch_idx] = -loss;

// then, we do reduction the result to calculate loss using 1 thread block
if (blockIdx.x > 0) return;

// cumulate workspace data
s_data[threadIdx.x] = 0.f;
for (int i = 0; i < batch_size; i += blockDim.x)
{
s_data[threadIdx.x] += workspace[threadIdx.x + i];
}

__syncthreads();

// reduction
for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
{
if (threadIdx.x + stride < batch_size)
s_data[threadIdx.x] += s_data[threadIdx.x + stride];

__syncthreads();
}

if (threadIdx.x == 0) {
reduced_loss[blockIdx.x] = s_data[0];
}
}