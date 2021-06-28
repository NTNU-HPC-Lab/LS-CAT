#include "includes.h"
/*
column
A[][] = ---------------------threadIdx.y
|
|
|
|
row      |
|
|
|
|
threadIdx.x
*/



#define TILE_WIDTH 16
#define TILE_WIDTH 16

#define ar 311
#define ac_br 312
#define bc 115

using namespace std;

__global__ void mat_mul(int *d_A, int *d_B, int *d_C, int rowA, int colA, int rowB, int colB, int rowC, int colC)
{
int row, col;
row = threadIdx.x + blockIdx.x*blockDim.x;      // 0 to rowA/rowC
col = threadIdx.y + blockIdx.y*blockDim.y;      // 0 to colB/colC

if(row < rowC && col < colC)
{
for(int i = 0; i < colA; i++)               // colA = rowB
d_C[row*colC + col] += d_A[row*colA + i]*d_B[i*colB + col];
}
}// End of mat_mul function