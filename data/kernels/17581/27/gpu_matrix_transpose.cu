#include "includes.h"
__global__ void gpu_matrix_transpose(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols)
{
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

if (idx < cols && idy < rows)
{
unsigned int pos = idy * cols + idx;
unsigned int trans_pos = idx * rows + idy;
mat_out[trans_pos] = mat_in[pos];
}
}