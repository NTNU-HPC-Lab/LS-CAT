#include "includes.h"
__global__ void computeMoment(int *readArr, int *writeArr, double *weightArr, int n){
int row = blockIdx.x*blockDim.x + threadIdx.x;
int col = blockIdx.y*blockDim.y + threadIdx.y;

// If coordinates are between boundaries
// update the write array accordingly
if(row < 517 && col < 517){
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
if 		(influence<-diff)	writeArr[row*n + col] = -1;
else if (influence>diff)	writeArr[row*n + col] = 1;
}
__syncthreads();

}