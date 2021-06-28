#include "includes.h"
__global__ void tileMatMul(float* matA, float* matB, float* matC, int aRows, int aCols, int bRows, int bCols, int cRows, int cCols)
{
//define row and column values
int Row = blockIdx.y * TILE_DIM + threadIdx.y;
int Col = blockIdx.x * TILE_DIM + threadIdx.x;

//shared memory arrays
__shared__ float sharedMatA[TILE_DIM][TILE_DIM];
__shared__ float sharedMatB[TILE_DIM][TILE_DIM];

float cResultValue = 0.0;

//calculate tiled matrix multiplication on shared memory
for(int i = 0; i < (aCols-1)/TILE_DIM+1; ++i)
{
if(Row < aRows && i*TILE_DIM+threadIdx.x < aCols)
{
sharedMatA[threadIdx.y][threadIdx.x] = matA[Row*aCols + i*TILE_DIM+threadIdx.x];
}
else
sharedMatA[threadIdx.y][threadIdx.x] = 0.0;

if(Col < bCols && i*TILE_DIM+threadIdx.y < cRows)
sharedMatB[threadIdx.y][threadIdx.x] = matB[(i*TILE_DIM+threadIdx.y)*bCols+Col];
else
sharedMatB[threadIdx.y][threadIdx.x] = 0.0;

__syncthreads();

for(int j = 0; j < TILE_DIM; ++j)
cResultValue += sharedMatA[threadIdx.y][j] * sharedMatB[j][threadIdx.x];

__syncthreads();
}

//put the results in the result matrix
if(Row < cRows && Col < cCols)
matC[Row*cCols+Col] = cResultValue;

}