#include "includes.h"
__global__ void gpu_matrix_mult_one(int *a, int *b, int *c, int m, int n, int k)
{
int row = blockIdx.y * blockDim.y + threadIdx.y; // get the row
int col = blockIdx.x * blockDim.x + threadIdx.x; // get the column
int sum = 0; // initialize the sum

if( col < k && row < m) // check to make sure that the thread needs to compute
{
for(int i = 0; i < n; i++)
{
sum += a[row * n + i] * b[i * k + col];
}
c[row * k + col] = sum;
}
}