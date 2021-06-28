#include "includes.h"
__global__ void matrixMultiply(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
//@@ Insert code to implement matrix multiplication here
__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;

int Row = by *TILE_WIDTH  + ty;
int Col = bx * TILE_WIDTH + tx;
//int Row = blockIdx.y*blockDim.y+threadIdx.y;
//int Col = blockIdx.x*blockDim.x+threadIdx.x;
float Cvalue = 0;

// Loop over the A and B tiles required to compute the C element
for (int t = 0; t < (numBRows-1)/TILE_WIDTH + 1; ++t)
{
if(Row < numARows && t*TILE_WIDTH+tx < numBRows)
{
// Collaborative loading of A
ds_A[ty][tx] = A[Row*numAColumns + t*TILE_WIDTH+tx];
}
else
{	// Control divergence at the edge
ds_A[ty][tx]= 0.0;
}

if ( t*TILE_WIDTH+ty < numBRows && Col < numBColumns)
{
// Collaborative loading of B if within range of matrix
ds_B[ty][tx] = B[(t*TILE_WIDTH+ty)*numBColumns + Col];
}
else
{
ds_B[ty][tx] = 0.0;
}

__syncthreads();

for (int i = 0; i < TILE_WIDTH; ++i)
{
Cvalue += ds_A[ty][i] * ds_B[i][tx];
}
__syncthreads();
}
if ( Row < numARows && Col < numBColumns)
C[Row*numBColumns+Col] = Cvalue;


}