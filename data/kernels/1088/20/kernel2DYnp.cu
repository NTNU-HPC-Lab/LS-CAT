#include "includes.h"
__global__ void kernel2DYnp ( double* dataNew, double* dataOld, double* boundaryTop, double* boundaryBottom, const double* weights, const int numSten, const int numStenTop, const int numStenBottom, const int nxLocal, const int nyLocal, const int BLOCK_Y, const int nx, const int nyTile, const int tileTop, const int tileBottom )
{
// Allocate the shared memory
extern __shared__ int memory[];

double* arrayLocal = (double*)&memory;
double* weigthsLocal = (double*)&arrayLocal[nxLocal * nyLocal];

// Move the weigths into shared memory
#pragma unroll
for (int k = 0; k < numSten; k++)
{
weigthsLocal[k] = weights[k];
}

// True matrix index
int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
int globalIdy = blockDim.y * blockIdx.y + threadIdx.y;

// Local matrix index
int localIdx = threadIdx.x;
int localIdy = threadIdx.y + numStenTop;

// Local sum variable
double sum = 0.0;

// Set index for summing stencil
int stenSet;

// Set all interior blocks
if (blockIdx.y != 0 && blockIdx.y != nyTile / (BLOCK_Y) - 1)
{
arrayLocal[localIdy * nxLocal + localIdx] = dataOld[globalIdy * nx + globalIdx];

if (threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + localIdx] = dataOld[(globalIdy - numStenTop) * nx + globalIdx];
}

if (threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataOld[(globalIdy + BLOCK_Y) * nx + globalIdx];
}

__syncthreads();


stenSet = threadIdx.y * nxLocal + threadIdx.x;

#pragma unroll
for (int k = 0; k < numSten; k++)
{
sum += weigthsLocal[k] * arrayLocal[stenSet + k * nxLocal];
}

__syncthreads();

dataNew[globalIdy * nx + globalIdx] = sum;
}

// Set all top boundary blocks
if (blockIdx.y == 0)
{
if (tileTop != 1)
{
arrayLocal[localIdy * nxLocal + localIdx] = dataOld[globalIdy * nx + globalIdx];

if (threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + localIdx] = boundaryTop[threadIdx.y * nx + globalIdx];
}

if (threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataOld[(globalIdy + BLOCK_Y) * nx + globalIdx];
}

__syncthreads();

stenSet = threadIdx.y * nxLocal + threadIdx.x;

#pragma unroll
for (int k = 0; k < numSten; k++)
{
sum += weigthsLocal[k] * arrayLocal[stenSet + k * nxLocal];
}

__syncthreads();

dataNew[globalIdy * nx + globalIdx] = sum;
}
else
{
arrayLocal[threadIdx.y * nxLocal + localIdx] = dataOld[globalIdy * nx + globalIdx];

if (threadIdx.y < numStenBottom)
{
arrayLocal[(threadIdx.y + BLOCK_Y) * nxLocal + localIdx] = dataOld[(globalIdy + BLOCK_Y) * nx + globalIdx];
}

__syncthreads();

stenSet = threadIdx.y * nxLocal + threadIdx.x;

#pragma unroll
for (int k = 0; k < numSten; k++)
{
sum += weigthsLocal[k] * arrayLocal[stenSet + k * nxLocal];
}

__syncthreads();

if (threadIdx.y < BLOCK_Y - numStenTop)
{
dataNew[(globalIdy + numStenTop) * nx + globalIdx] = sum;
}
}
}


// Set the bottom boundary blocks
if (blockIdx.y == nyTile / BLOCK_Y - 1)
{
if (tileBottom != 1)
{
arrayLocal[localIdy * nxLocal + localIdx] = dataOld[globalIdy * nx + globalIdx];

if (threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + localIdx] = dataOld[(globalIdy - numStenTop) * nx + globalIdx];
}

if (threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = boundaryBottom[threadIdx.y * nx + globalIdx];
}

__syncthreads();

stenSet = threadIdx.y * nxLocal + threadIdx.x;

#pragma unroll
for (int k = 0; k < numSten; k++)
{
sum += weigthsLocal[k] * arrayLocal[stenSet + k * nxLocal];
}

__syncthreads();

dataNew[globalIdy * nx + globalIdx] = sum;
}
else
{
arrayLocal[localIdy * nxLocal + localIdx] = dataOld[globalIdy * nx + globalIdx];

if (threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + localIdx] = dataOld[(globalIdy - numStenTop) * nx + globalIdx];
}

stenSet = threadIdx.y * nxLocal + threadIdx.x;

#pragma unroll
for (int k = 0; k < numSten; k++)
{
sum += weigthsLocal[k] * arrayLocal[stenSet + k * nxLocal];
}

__syncthreads();

if (threadIdx.y < BLOCK_Y - numStenBottom)
{
dataNew[globalIdy * nx + globalIdx] = sum;
}
}
}
}