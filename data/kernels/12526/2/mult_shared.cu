#include "includes.h"
__global__ void mult_shared( int *A, int *B, int *result, int n)
{	int k;
int kk;
const int bx = BLOCK_X, by = BLOCK_Y;
const int col = blockIdx.x*bx + threadIdx.x;
const int row = blockIdx.y*by + threadIdx.y;

__shared__ int a[BLOCK_X][BLOCK_Y] , b[BLOCK_X][BLOCK_Y];
if ((col < n) && (row < n))
{
int c = 0;
for (k=0; k < n; k++)
{
a[threadIdx.x][threadIdx.y] = A[ col * n + k*by + threadIdx.y];
b[threadIdx.y][threadIdx.x] = B[ row + n * (k*bx+threadIdx.x)];
__syncthreads(); // Synchronizes all threads in a block
for (kk=0; kk< bx; kk++)
c += a[kk][threadIdx.x]*b[kk][threadIdx.y];
__syncthreads(); // Avoids memory hazards
}
result[col*n+row] = c;
}

}