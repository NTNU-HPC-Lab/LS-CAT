#include "includes.h"
__global__ void cunn_CriterionFilter_updateOutput_kernel( float *target, float *ignored_label, int bound, int batch_size, int map_nelem, int blocks_per_sample)
{
int i;
int sample = blockIdx.x / blocks_per_sample;
int step = blockDim.x * blocks_per_sample;
int toffset = sample * map_nelem;
int ignored_label_num = (int)(ignored_label[0]);
for (i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x; i < map_nelem; i += step) {
if (target[toffset + i] == ignored_label_num) {
target[toffset + i] = (float) bound + 1;
}
}
}