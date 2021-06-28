#include "includes.h"
__global__ void kernel2DXp ( double* dataOutput, double* dataInput, const double* weights, const int numSten, const int numStenLeft, const int numStenRight, const int nxLocal, const int nyLocal, const int BLOCK_X, const int nx )
{
// -----------------------------
// Allocate the shared memory
// -----------------------------

extern __shared__ int memory[];

double* arrayLocal = (double*)&memory;
double* weigthsLocal = (double*)&arrayLocal[nxLocal * nyLocal];

// Move the weigths into shared memory
#pragma unroll
for (int k = 0; k < numSten; k++)
{
weigthsLocal[k] = weights[k];
}

// -----------------------------
// Set the indexing
// -----------------------------

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

// -----------------------------
// Set interior
// -----------------------------

arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

// -----------------------------
// Set x boundaries
// -----------------------------

// If block is in the interior
if (blockIdx.x != 0 && blockIdx.x != nx / BLOCK_X - 1)
{

if (threadIdx.x < numStenLeft)
{
arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (globalIdx - numStenLeft)];
}

if (threadIdx.x < numStenRight)
{
arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
}
}

// If block is on the left boundary
if (blockIdx.x == 0)
{
arrayLocal[localIdy * nxLocal + localIdx] = dataInput[globalIdy * nx + globalIdx];

if (threadIdx.x < numStenLeft)
{
arrayLocal[localIdy * nxLocal + threadIdx.x] = dataInput[globalIdy * nx + (nx - numStenLeft + threadIdx.x)];
}

if (threadIdx.x < numStenRight)
{
arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + globalIdx + BLOCK_X];
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

if (threadIdx.x < numStenRight)
{
arrayLocal[localIdy * nxLocal + (localIdx + BLOCK_X)] = dataInput[globalIdy * nx + threadIdx.x];
}
}

// -----------------------------
// Compute the stencil
// -----------------------------

__syncthreads();

stenSet = localIdy * nxLocal + threadIdx.x;

#pragma unroll
for (int k = 0; k < numSten; k++)
{
sum += weigthsLocal[k] * arrayLocal[stenSet + k];
}

__syncthreads();

// -----------------------------
// Copy back to global
// -----------------------------

dataOutput[globalIdy * nx + globalIdx] = sum;
}