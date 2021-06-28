#include "includes.h"
__global__ void cunn_CriterionFilter_updateGradInput_kernel( float *gradInput, float *target, float *ignored_label, int batch_size, int n_classes, int map_nelem, int blocks_per_sample)
{
int i, t;
int sample = blockIdx.x / blocks_per_sample;
int step = blockDim.x * blocks_per_sample;
int toffset = sample * map_nelem;
int ioffset = sample * map_nelem * n_classes;
int ignored_label_num = (int)(ignored_label[0]);
for (i = (blockIdx.x % blocks_per_sample) * blockDim.x + threadIdx.x; i < map_nelem; i += step) {
t = (int)target[toffset + i];
if (t == ignored_label_num) {
int j;
for (j = 0; j < n_classes; j++) gradInput[ioffset + j * map_nelem + i] = 0;
}
}
}