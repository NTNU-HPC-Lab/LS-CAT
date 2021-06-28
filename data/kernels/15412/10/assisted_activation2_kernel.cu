#include "includes.h"
__global__ void assisted_activation2_kernel(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int xy = i % size;
int b = i / size;
float beta = 1 - alpha;

if (b < batches) {
for (int c = 0; c < channels; ++c) {
if(gt_gpu[i] == 0)
output[xy + size*(c + channels*b)] *= beta;

}
}
}