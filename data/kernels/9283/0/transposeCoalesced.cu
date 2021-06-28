#include "includes.h"
/*
* This file is an attempt at producing what the generated target code
* should look like for the multiplyMatrixMatrix routine.
*/

/* Prototype matrix representation. */
struct dag_array_t{
size_t rows;
size_t cols;
int* matrix;
};

/*
DAG Primitive. Here, we leverage the NVIDIA developer examples
to obtain a high-bandwith operation. They make use of shared memory
to avoid strided global memory accesses, and instead perform the
strided access in the shared block, which is roughly a ~3x improvement.

TILE_DIM = 32
BLOCK_ROWS = 8

https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
*/
const int tp_TILE_DIM = 32;
const int tp_BLOCK_ROWS = 8;




// We use single-dimensional lists.
__global__ void transposeCoalesced(int *result, const int *in)
{
const int TILE_DIM = tp_TILE_DIM;
const int BLOCK_ROWS = tp_BLOCK_ROWS;

__shared__ int tile[TILE_DIM][TILE_DIM];

int x = blockIdx.x * TILE_DIM + threadIdx.x;
int y = blockIdx.y * TILE_DIM + threadIdx.y;
int width = gridDim.x * TILE_DIM;

for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*width + x];

__syncthreads();

x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
y = blockIdx.x * TILE_DIM + threadIdx.y;

for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
result[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}