#include "includes.h"
__global__ void matrix_mul_shared(float *ad,float *bd,float *cd,int N)
{
float pvalue=0;
int TILE=blockDim.x;
int ty=threadIdx.y;
int tx=threadIdx.x;

//allocate shared memory per block
__shared__ float ads[16][16];
__shared__ float bds[16][16];

//find Row and Column corresponding to a data element for each thread
int Row = blockIdx.y * blockDim.y + threadIdx.y;
int Col = blockIdx.x * blockDim.x + threadIdx.x;

//iterate through TILEs to traverse whole WIDTH
for(int i=0;i< N/TILE;++i)
{
//copy values of data TILE into shared memory
ads[ty][tx] = ad[Row * N + (i * TILE) + tx];
bds[ty][tx] = bd[(i * TILE + ty) * N + Col];

__syncthreads();                            //synchronize to confirm that whole TILE has been copied

//calculate partial dot-product
for(int k=0;k<TILE;k++)
pvalue += ads[ty][k] * bds[k][tx];

__syncthreads();                            //synchronize to confirm that whole partial product corresponding to all threads of the block has been calculated
}

//store dot product at corresponding positon in resultant Matrix
cd[Row * N + Col] = pvalue;
}