#include "includes.h"
__global__ void assisted_activation_kernel(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int xy = i % size;
int b = i / size;

if (b < batches) {
for (int c = 0; c < channels; ++c) {
output[xy + size*(c + channels*b)] += alpha * gt_gpu[i] * a_avg_gpu[i];
//output[xy + size*(c + channels*b)] += gt_gpu[i] * a_avg_gpu[i];
//output[xy + size*(c + channels*b)] += gt_gpu[i] * output[xy + size*(c + channels*b)];
//output[xy + size*(c + channels*b)] = a_avg_gpu[i];
}
}
}