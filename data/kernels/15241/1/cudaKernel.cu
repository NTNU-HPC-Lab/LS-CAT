#include "includes.h"
__global__ void cudaKernel(int n, double* gpuWeights, int* gpuG, int* gpuTempGrid, int *flag)
{
// Moment's coordinates in the grid //
int momentCol = blockIdx.x*blockDim.x + threadIdx.x;
int	momentRow = blockIdx.y*blockDim.y + threadIdx.y;
// Shared memory allocated for weights //
__shared__ double sharedWeights[25];
// Shared memory allocated for a block of moments //
// Size is (BLOCK_SIZE+4)^2 //
int sharedSize = (BLOCK_SIZE+4);
__shared__ int sharedG[(BLOCK_SIZE+4)*(BLOCK_SIZE+4)];
// Moment's coordinates in the shared memory //
int sharedRow = threadIdx.y+2;
int sharedCol= threadIdx.x+2;

// Indexes used to read from global memory //
int idxRow, idxCol;
// Variable storing neighbourhood's influence //
double weightFactor = 0.0;
// Store weights in shared memory //
if(threadIdx.x<5 && threadIdx.y<5)
sharedWeights[threadIdx.x*5+threadIdx.y] = gpuWeights[threadIdx.x*5+threadIdx.y];

// In this double loop, moments and their necessary neighbours are //
// passed from global to shared memory. After this data trannsfer //
// each thread calculates the atomic spin of its moment, based on //
// the spins of the moment's neighbours  //
for(int i=momentRow; i<n+2; i+=blockDim.y*gridDim.y)
{
for(int j=momentCol; j<n+2; j+=blockDim.x*gridDim.x)
{
// Store moment in shared memory //
sharedG[sharedRow*sharedSize+sharedCol] = gpuG[( (i + n)%n )*n + ( (j + n)%n )];
// In this if statement, we also add to shared memory the -2 left neighbour, //
// the +BLOCK_SIZE right neighbour of every moment on the 0 and 1 column. //
//  We also add the corners of the block that are necessary in order //
// to calculate the atomic spins of the block. All this work is done  //
// by threads with positioned in 0 and 1 column. //
if(threadIdx.x < 2)
{
// Left Boundaries //
idxRow = (i + n)%n;
idxCol = (-2 + j + n)%n;
sharedG[(sharedRow)*sharedSize+sharedCol-2] = gpuG[n*idxRow+idxCol];
// Right Boundaries //
idxCol = (BLOCK_SIZE + j + n)%n;
sharedG[(sharedRow)*sharedSize+sharedCol+BLOCK_SIZE] = gpuG[n*idxRow+idxCol];

if(threadIdx.y <2)
{
// Top Left Corner //
idxRow = (-2 + i + n)%n;
idxCol = (-2 + j + n)%n;
sharedG[(sharedRow-2)*sharedSize+sharedCol-2] = gpuG[n*idxRow+idxCol];

// Bottom Left Corner //
idxRow = (i + n + BLOCK_SIZE)%n;
idxCol = (-2 + j + n)%n;
sharedG[(sharedRow+  BLOCK_SIZE)*sharedSize+sharedCol-2] = gpuG[n*idxRow+idxCol];

// Top Right Corner//
idxRow = (-2 + i + n)%n;
idxCol = (j + n + BLOCK_SIZE)%n;
sharedG[(sharedRow-2)*sharedSize+sharedCol + BLOCK_SIZE] = gpuG[n*idxRow+idxCol];

// Bottom Right Corner//
idxRow = (i + n+BLOCK_SIZE)%n;
idxCol = (j + n+BLOCK_SIZE)%n;
sharedG[(sharedRow+BLOCK_SIZE)*sharedSize+sharedCol+BLOCK_SIZE] = gpuG[n*idxRow+idxCol];
}
}
// In this if statement we also add the top and bottom neighbours of //
// the block. This is done by threads positioned in 0 and 1 row. //
if(threadIdx.y < 2)
{
// Top Boundaries //
idxRow = (-2 + i + n)%n;
idxCol = (j + n)%n;
sharedG[(sharedRow-2)*sharedSize+sharedCol] = gpuG[n*idxRow+idxCol];

// Bottom Boundaries //
idxRow = (i + n+BLOCK_SIZE)%n;
sharedG[(sharedRow+BLOCK_SIZE)*sharedSize+sharedCol] = gpuG[n*idxRow+idxCol];
}
// Synchronize all threads to ensure that writting to shared memory is done. //
__syncthreads();
// Compute the spins of moments with coordinates within n-size //
if(i<n && j<n)
{
weightFactor = 0.0;
for(int row=0; row<5; row++)
{
for(int col=0; col<5; col++)
{
if(col==2 && row==2)
continue;
//Calculate neighbourhood's total nfluence
weightFactor+= sharedG[(sharedRow-2+row)*sharedSize+sharedCol-2+col] * sharedWeights[row*5+col];
}
}
// Determine future atomic spin value based on total influence //
if(weightFactor < 0.0001 && weightFactor > -0.0001)
{
gpuTempGrid[n*i+j] = sharedG[sharedRow*sharedSize+sharedCol];
}else if(weightFactor > 0.00001)
{
gpuTempGrid[n*i+j] = 1;
if (gpuG[n*i+j] == -1)
{
*flag = 1;
}
}else
{
gpuTempGrid[n*i+j] = -1;
if (gpuG[n*i+j] == -1)
{
*flag = 1;
}
}

}
// Synchronize threads before writting again to shared memory //
// to ensure that no one is reading data from shared memory //
__syncthreads();

}
}
}