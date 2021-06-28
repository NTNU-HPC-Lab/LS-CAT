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
__global__ void multiplyMatrixVector(int* result, int* matrix, int* vector, int cols)
{
__shared__ int reduce_array[256]; // Within a block

int vector_slice_offset = blockIdx.x * cols + threadIdx.x;
int matrix_slice_offset = blockIdx.y * cols + threadIdx.x;
reduce_array[threadIdx.x] = matrix[matrix_slice_offset] * vector[vector_slice_offset];

__syncthreads();

// Sequential reduce.
if (threadIdx.x == 0){
int accumulator = 0;
for (int i = 0; i < blockDim.x; i++)
{
accumulator += reduce_array[i];
}
result[blockIdx.x * cols + blockIdx.y] = accumulator;
}
}