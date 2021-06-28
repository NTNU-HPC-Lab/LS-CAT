#include "includes.h"
__global__ void set_scales_dropblock_kernel(float *drop_blocks_scale, int block_size_w, int block_size_h, int outputs, int batch)
{
const int index = blockIdx.x*blockDim.x + threadIdx.x;
if (index >= batch) return;

//printf(" drop_blocks_scale[index] = %f \n", drop_blocks_scale[index]);
const float prob = drop_blocks_scale[index] / (float)outputs;
const float scale = 1.0f / (1.0f - prob);
drop_blocks_scale[index] = scale;
}