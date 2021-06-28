#include "includes.h"
__global__ void computeMoment(int8_t *readArr, int8_t *writeArr, float *weightArr, int n, int tileSize){
int row_init = blockIdx.x*(blockDim.x*tileSize) + threadIdx.x*tileSize;
int col_init = blockIdx.y*(blockDim.y*tileSize) + threadIdx.y*tileSize;

// Assign each thread a tileSizeXtileSize tile
for(int ii=0; ii<tileSize; ++ii){
for (int jj=0; jj<tileSize; ++jj){
int row = row_init + ii;
int col = col_init + jj;

// If coordinates are between boundaries
// update the write array accordingly
if(row < n && col < n){
float influence = 0.0f;
for (int i=-2; i<3; i++)
{
for (int j=-2; j<3; j++)
{
//add extra n so that modulo behaves like mathematics modulo
//that is return only positive values
int y = (row+i+n)%n;
int x = (col+j+n)%n;
influence += weightArr[i*5 + j]*readArr[y*n + x];
}
}

writeArr[row*n + col] = readArr[row*n + col];
if 	(influence<-diff)	writeArr[row*n + col] = -1;
else if (influence>diff)	writeArr[row*n + col] = 1;
__syncthreads();
}
}
}
}