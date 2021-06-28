#include "includes.h"
__global__ void cudaKernel(int n, double* gpuWeights, int* gpuG, int* gpuTempGrid, int *flag)
{
// Moment's coordinates in the grid //
// allocate shared memory for weights
int momentCol = blockIdx.x*blockDim.x + threadIdx.x;
int	momentRow = blockIdx.y*blockDim.y + threadIdx.y;

int gridRowIdx, gridColIdx;
// Variable storing the total neighbourhood influence //
double weightFactor = 0.0;
// Each thread calculates the spin for a block of moments //
// The step is based on the GRID_SIZE and BLOCK_SIZE //
for(int i=momentRow; i<n; i+=blockDim.y*gridDim.y)
{
for(int j=momentCol; j<n; j+=blockDim.x*gridDim.x)
{
weightFactor = 0.0;
// Read 24 neighbours of every moment and calculate their total influence //
for(int weightsRow=0; weightsRow<5; weightsRow++)
{
for(int weightsCol=0; weightsCol<5; weightsCol++)
{
if(weightsCol==2 && weightsRow==2)
continue;
// Calculate neighbour's coordinates in G //
// using modulus to satisfy boundary conditions //
gridRowIdx = (weightsRow - 2 + i + n) % n;
gridColIdx = (weightsCol - 2 + j + n) % n;

weightFactor+= gpuG[gridRowIdx * n + gridColIdx] * gpuWeights[weightsRow*5+weightsCol];
}
}
// Update moment's atomic spin //
// Set flag if a spin value transition has been done //
if(weightFactor < 0.0001 && weightFactor > -0.0001)
{
gpuTempGrid[n*i+j] = gpuG[n*i+j];
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
}
}