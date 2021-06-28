#include "includes.h"
__global__ void kernel2DXnp ( double* dataOutput, double* dataInput, const double* weights, const int numSten, const int numStenLeft, const int numStenRight, const int nxLocal, const int nyLocal, const int BLOCK_X, const int nx )
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
int localIdx = threadIdx.x + numStenLeft;
int localIdy = threadIdx.y;

// Local sum variable
double sum = 0.0;

// Set index for summing stencil
int stenSet;

// Set all interior blocks
if (blockIdx.x != 0 && blockIdx.x != nx / (BLOCK_X) - 1)
{
arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

if (threadIdx.x < numStenLeft)
{
arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
}

if (threadIdx.x < numStenRight)
{
arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
}

__syncthreads();


stenSet = localIdy * nxLocal + threadIdx.x;

#pragma unroll
for (int k = 0; k < numSten; k++)
{
sum += weigthsLocal[k] * arrayLocal[stenSet + k];
}

dataOutput[globalIdy * nx + globalIdx] = sum;
}

// Set all left boundary blocks
if (blockIdx.x == 0)
{
arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + globalIdx];

if (threadIdx.x < numStenRight)
{
arrayLocal[localIdy * nxLocal + threadIdx.x + BLOCK_X] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
}

__syncthreads();

if (threadIdx.x >= numStenLeft)
{

stenSet = localIdy * nxLocal + threadIdx.x - numStenLeft;

#pragma unroll
for (int k = 0; k < numSten; k++)
{
sum += weigthsLocal[k] * arrayLocal[stenSet + k];
}

dataOutput[globalIdy * nx + globalIdx] = sum;
}
}

// Set the right boundary blocks
if (blockIdx.x == nx / BLOCK_X - 1)
{
arrayLocal[localIdy * nxLocal + threadIdx.x + numStenLeft] = dataInput[globalIdy * nx + globalIdx];

if (threadIdx.x < numStenLeft)
{
arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
}

__syncthreads();

if (threadIdx.x < BLOCK_X - numStenRight)
{

stenSet = localIdy * nxLocal + threadIdx.x;

#pragma unroll
for (int k = 0; k < numSten; k++)
{
sum += weigthsLocal[k] * arrayLocal[stenSet + k];
}
}

dataOutput[globalIdy * nx + globalIdx] = sum;
}
}