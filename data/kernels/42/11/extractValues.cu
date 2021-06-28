#include "includes.h"
__global__ void extractValues(void* fb, int* voxels, int num_voxels, int* values) {
int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < num_voxels) {
//TODO: Make this support other storage_type's besides int32
float* tile = (float*)fb;
values[index] = __float_as_int(tile[voxels[index]]);
}
}