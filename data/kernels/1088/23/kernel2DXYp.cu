#include "includes.h"
__global__ void kernel2DXYp ( double* dataOutput, double* dataInput, double* boundaryTop, double* boundaryBottom, const double* weights, const int numSten, const int numStenHoriz, const int numStenLeft, const int numStenRight, const int numStenVert, const int numStenTop, const int numStenBottom, const int nxLocal, const int nyLocal, const int BLOCK_X, const int BLOCK_Y, const int nx, const int nyTile )
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
int localIdy = threadIdx.y + numStenTop;

// Local sum variable
double sum = 0.0;

// Set index for summing stencil
int stenSet;

// Set temporary index for looping
int temp;

// Use to loop over indexing in the weighsLocal
int weight = 0;

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
// Set y boundaries
// -----------------------------

// Set interior y boundary
if (blockIdx.y != 0 && blockIdx.y != nyTile / BLOCK_Y - 1)
{
if (threadIdx.y < numStenTop )
{
arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
}

if (threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
}
}

// Set top y boundary
if (blockIdx.y == 0)
{
if (threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + localIdx] = boundaryTop[threadIdx.y * nx + globalIdx];
}

if (threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = dataInput[(globalIdy + BLOCK_Y) * nx + globalIdx];
}
}

// Set bottom y boundary
if (blockIdx.y == nyTile / BLOCK_Y - 1)
{
if (threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + localIdx] = dataInput[(globalIdy - numStenTop) * nx + globalIdx];
}

if (threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + localIdx] = boundaryBottom[threadIdx.y * nx + globalIdx];
}
}

// -----------------------------
// Corners - Interior of tile
// -----------------------------

// Set interior y boundary
if (blockIdx.y != 0 && blockIdx.y != nyTile / BLOCK_Y - 1)
{
// If block is in the interior
if (blockIdx.x != 0 && blockIdx.x != nx / BLOCK_X - 1)
{
// Top Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
}

// Top Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
}

// Bottom Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
}

// Bottom Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
}
}

// If block is on the left boundary
if (blockIdx.x == 0)
{
// Top Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (nx - numStenLeft + threadIdx.x)];
}

// Top Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
}

// Bottom Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (nx - numStenLeft + threadIdx.x)];
}

// Bottom Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
}
}

// If block is on the right boundary
if (blockIdx.x == nx / BLOCK_X - 1)
{
// Top Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
}

// Top Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + threadIdx.x];
}

// Bottom Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
}

// Bottom Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + threadIdx.x];
}
}
}

// -----------------------------
// Corners - Top of tile
// -----------------------------

// Set top y boundary
if (blockIdx.y == 0)
{
// If block is in the interior
if (blockIdx.x != 0 && blockIdx.x != nx / BLOCK_X - 1)
{
// Top Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nx + (globalIdx - numStenLeft)];
}

// Top Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nx + (globalIdx + BLOCK_X)];
}

// Bottom Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
}

// Bottom Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
}
}

// If block is on the left boundary
if (blockIdx.x == 0)
{
// Top Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nx + (nx - numStenLeft + threadIdx.x)];

}

// Top Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nx + (globalIdx + BLOCK_X)];
}

// Bottom Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (nx - numStenLeft + threadIdx.x)];
}

// Bottom Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx + BLOCK_X)];
}
}

// If block is on the right boundary
if (blockIdx.x == nx / BLOCK_X - 1)
{
// Top Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = boundaryTop[threadIdx.y * nx + (globalIdx - numStenLeft)];
}

// Top Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = boundaryTop[threadIdx.y * nx + threadIdx.x];
}

// Bottom Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = dataInput[(globalIdy + BLOCK_Y) * nx + (globalIdx - numStenLeft)];
}

// Bottom Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy + BLOCK_Y) * nx + threadIdx.x];
}
}
}

// -----------------------------
// Corners - Bottom of tile
// -----------------------------

// Set bottom y boundary
if (blockIdx.y == nyTile / BLOCK_Y - 1)
{
// If block is in the interior
if (blockIdx.x != 0 && blockIdx.x != nx / BLOCK_X - 1)
{
// Top Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
}

// Top Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
}

// Bottom Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nx + (globalIdx - numStenLeft)];
}

// Bottom Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nx + (globalIdx + BLOCK_X)];

}
}

// If block is on the left boundary
if (blockIdx.x == 0)
{
// Top Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (nx - numStenLeft + threadIdx.x)];
}

// Top Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx + BLOCK_X)];
}

// Bottom Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nx + (nx - numStenLeft + threadIdx.x)];
}

// Bottom Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nx + (globalIdx + BLOCK_X)];
}
}

// If block is on the right boundary
if (blockIdx.x == nx / BLOCK_X - 1)
{
// Top Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + threadIdx.x] = dataInput[(globalIdy - numStenTop) * nx + (globalIdx - numStenLeft)];
}

// Top Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenTop)
{
arrayLocal[threadIdx.y * nxLocal + (localIdx + BLOCK_X)] = dataInput[(globalIdy - numStenTop) * nx + threadIdx.x];
}

// Bottom Left
if (threadIdx.x < numStenLeft && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + threadIdx.x] = boundaryBottom[threadIdx.y * nx + (globalIdx - numStenLeft)];
}

// Bottom Right
if (threadIdx.x < numStenRight && threadIdx.y < numStenBottom)
{
arrayLocal[(localIdy + BLOCK_Y) * nxLocal + (localIdx + BLOCK_X)] =  boundaryBottom[threadIdx.y * nx + threadIdx.x];
}
}
}

// -----------------------------
// Compute the stencil
// -----------------------------

__syncthreads();

stenSet = (localIdy - numStenTop) * nxLocal + (localIdx - numStenLeft);
weight = 0;

for (int j = 0; j < numStenVert; j++) // Allow for the point we're actually at
{
temp = j * nxLocal;

for (int i = 0; i < numStenHoriz; i++) // Allow for the point we're actually at
{
sum += weigthsLocal[weight] * arrayLocal[stenSet + temp + i];

weight++;
}
}

__syncthreads();

// -----------------------------
// Copy back to global
// -----------------------------

dataOutput[globalIdy * nx + globalIdx] = sum;

}