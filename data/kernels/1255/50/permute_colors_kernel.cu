#include "includes.h"
__global__ void permute_colors_kernel(int num_rows, int *row_colors, int *color_permutation)
{
int row_id = blockIdx.x * blockDim.x + threadIdx.x;

for ( ; row_id < num_rows ; row_id += blockDim.x * gridDim.x )
{
int color = row_colors[row_id];
#if __CUDA_ARCH__ >= 350
color = __ldg(color_permutation + color);
#else
color = color_permutation[color];
#endif
row_colors[row_id] = color;
}
}