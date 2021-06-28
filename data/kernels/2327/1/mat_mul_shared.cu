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

__global__ void mat_mul_shared(int *d_A, int *d_B, int *d_C, int rowA, int colA, int rowB, int colB, int rowC, int colC)
{
int bx = blockIdx.x,     by = blockIdx.y;
int tx = threadIdx.x,    ty = threadIdx.y;
int row = tx + bx*TILE_WIDTH;      // 0 to rowA/rowC
int col = ty + by*TILE_WIDTH;      // 0 to colB/colC

__shared__ int s_A[TILE_WIDTH][TILE_WIDTH], s_B[TILE_WIDTH][TILE_WIDTH];
int cvalue = 0;

for(int i = 0; i < (colA+TILE_WIDTH-1)/TILE_WIDTH; i++)
{
if(row < rowA && i*TILE_WIDTH+ty < colA)
s_A[tx][ty] = d_A[row*colA + i*TILE_WIDTH+ty];
else
s_A[tx][ty] = 0;

if(i*TILE_WIDTH+tx < rowB && col < colB)
s_B[tx][ty] = d_B[(i*TILE_WIDTH+tx)*colB + col];
else
s_B[tx][ty] = 0;

__syncthreads();

for(int k = 0; k < TILE_WIDTH; k++)
cvalue += s_A[tx][k]*s_B[k][ty];

__syncthreads();
}

if(row < rowC && col < colC)
d_C[row*colC + col] = cvalue;

}// End of mat_mul_shared function