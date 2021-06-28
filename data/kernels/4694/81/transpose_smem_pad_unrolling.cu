#include "includes.h"
__global__ void transpose_smem_pad_unrolling(int * in, int* out, int nx, int ny)
{
__shared__ int tile[BDIMY * (2 * BDIMX + IPAD)];

//input index
int ix, iy, in_index;

//output index
int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

//ix and iy calculation for input index
ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
iy = blockDim.y * blockIdx.y + threadIdx.y;

//input index
in_index = iy * nx + ix;

//1D index calculation fro shared memory
_1d_index = threadIdx.y * blockDim.x + threadIdx.x;

//col major row and col index calcuation
i_row = _1d_index / blockDim.y;
i_col = _1d_index % blockDim.y;

//coordinate for transpose matrix
out_ix = blockIdx.y * blockDim.y + i_col;
out_iy = 2 * blockIdx.x * blockDim.x + i_row;

//output array access in row major format
out_index = out_iy * ny + out_ix;

if (ix < nx && iy < ny)
{
int row_idx = threadIdx.y * (2 * blockDim.x + IPAD) + threadIdx.x;

//load from in array in row major and store to shared memory in row major
tile[row_idx] = in[in_index];
tile[row_idx+ BDIMX] = in[in_index + BDIMX];

//wait untill all the threads load the values
__syncthreads();

int col_idx = i_col * (2 * blockDim.x + IPAD) + i_row;

out[out_index] = tile[col_idx];
out[out_index + ny* BDIMX] = tile[col_idx + BDIMX];
}
}