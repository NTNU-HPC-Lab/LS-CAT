#include "includes.h"
/*
*  file name: mm_omp_vs_cuda.cu
*
*  mm_omp_vs_cuda.cu contains the code that realize some common used matrix operations in CUDA, and
*  an implementation of matrix multiplication speedup via openmp, this is a practice to compare the
*  of performance of cuda and openmp, as well as a trail of using cuda and openmp in the same program
*
*  this is a toy program for learning CUDA, some functions are reusable in other project
*  note:
*       compile: nvcc -Xcompiler \-fopenmp -lgomp mm_omp_vs_cuda.cu
*/
#define BLOCK_SIZE 16

/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters:
&a GPU device pointer to a m X n matrix (A)
&b GPU device pointer to a n X k matrix (B)
&c GPU device output purpose pointer to a m X k matrix (C)
to store the result

Note:
grid and block should be configured as:
dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/

/*
*********************************************************************
function name: cpu_matrix_mult

description: dot product of two matrix (not only square) in CPU,
for validating GPU results

parameters:
&a CPU device pointer to a n X n matrix (A)
&b CPU device pointer to a n X n matrix (B)
&c CPU device output purpose pointer to a n X n matrix (C)
to store the result
Note:
grid and block should be configured as:

dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/

/*
*********************************************************************
function name: gpu_matrix_transpose

description: matrix transpose

parameters:
&mat_in GPU device pointer to a rows X cols matrix
&mat_out GPU device output purpose pointer to a cols X rows matrix
to store the result
Note:
grid and block should be configured as:
dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
/*
*********************************************************************
function name: cpu_matrix_mult

description: dot product of two matrix (not only square) in CPU,
for validating GPU results

parameters:
&a CPU host pointer to a m X n matrix (A)
&b CPU host pointer to a n X k matrix (B)
&c CPU host output purpose pointer to a m X k matrix (C)
to store the result
return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n)
{
__shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
__shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
int tmp = 0;
int idx;

for (int sub = 0; sub < gridDim.x; ++sub)
{
idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
if(idx >= n*n)
{
// n may not divisible by BLOCK_SIZE
tile_a[threadIdx.y][threadIdx.x] = 0;
}
else
{
tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
}

idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
if(idx >= n*n)
{
tile_b[threadIdx.y][threadIdx.x] = 0;
}
else
{
tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
}
__syncthreads();

for (int k = 0; k < BLOCK_SIZE; ++k)
{
tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
}
__syncthreads();
}
if(row < n && col < n)
{
d_result[row * n + col] = tmp;
}
}