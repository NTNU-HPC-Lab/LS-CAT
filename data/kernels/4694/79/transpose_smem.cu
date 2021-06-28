#include "includes.h"
__global__ void transpose_smem(int * in, int* out, int nx, int ny)
{
__shared__ int tile[BDIMY][BDIMX];

//input index
int ix, iy, in_index;

//output index
int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

//ix and iy calculation for input index
ix = blockDim.x * blockIdx.x + threadIdx.x;
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
out_iy = blockIdx.x * blockDim.x + i_row;

//output array access in row major format
out_index = out_iy * ny + out_ix;

if (ix < nx && iy < ny)
{
//load from in array in row major and store to shared memory in row major
tile[threadIdx.y][threadIdx.x] = in[in_index];

//wait untill all the threads load the values
__syncthreads();

out[out_index] = tile[i_col][i_row];
}
}