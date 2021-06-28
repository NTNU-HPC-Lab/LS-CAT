#include "includes.h"
__device__ int greatest_row;  __device__  void swap(float* arr, int ind_a, int ind_b)
{
float tmp = arr[ind_a];
arr[ind_a] = arr[ind_b];
arr[ind_b] = tmp;
}
__global__ void swapRow(float* mat, float* b, int cols, int num_block, int k)
{
int row_i = greatest_row;
if (k != row_i) //If the same row don't swap.
{
int row_k = k*cols;
int swap_row = row_i*cols;
//	Calc. swap interval
int i = threadIdx.x + blockIdx.x * blockDim.x;
// Swap matrix
for (; i < cols; i += num_block*blockDim.x)
swap(mat, swap_row + i, row_k + i);
// Swap b
if(blockIdx.x == 0 && threadIdx.x == 0)
swap(b, row_i, k);
}
}