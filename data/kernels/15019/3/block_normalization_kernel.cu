#include "includes.h"
__global__ void block_normalization_kernel(float* histograms, float* descriptor, int histograms_step, int block_grid_width, int block_grid_height, int block_width, int block_height, int num_bins, int cell_grid_width, int block_stride_x, int block_stride_y)
{
//TODO: make the buffer sizes dependent on an input or template parameter.
// Each thread block will process 8 hog blocks. Each hog block has 4 cells.
// Each cell has 9 bins.
__shared__ float s_blocks[9 * 4 * 8];
__shared__ float L1_norm[8];
int block_x = blockIdx.x * 8 + threadIdx.z;
if(block_x >= block_grid_width)
{
return;
}
int block_y = blockIdx.y;
if(block_y >= block_grid_height)
{
return;
}
int block_idx = block_y * blockDim.y + block_x;
int cell_x = block_x * block_stride_x + threadIdx.y % 2;
int cell_y = block_y * block_stride_y + threadIdx.y / 2;
int hist_idx = histograms_step * cell_y + num_bins * (cell_x) + threadIdx.x;

int s_blocks_idx = 9 * threadIdx.y + threadIdx.x;
s_blocks[s_blocks_idx] = histograms[hist_idx];

__syncthreads();

int thread_id = 36 * threadIdx.z + 9 * threadIdx.y + threadIdx.x;
int elements_per_block = block_height * block_width * num_bins;
if(thread_id < 8)
{
L1_norm[thread_id] = 0.0f;
for(int i = 0; i < elements_per_block; ++i)
{
L1_norm[thread_id] += s_blocks[elements_per_block * thread_id + i];
}
}

__syncthreads();

descriptor[elements_per_block * block_idx + s_blocks_idx] =
s_blocks[s_blocks_idx] / L1_norm[threadIdx.z];
}