#include "includes.h"
__global__ void ComputeSquareDistance(float* dOut, float* dIn, int n, int d)
{
// Load values that will be reused
__shared__ float blockA[KNN_BLOCK_SIZE][KNN_BLOCK_SIZE];
__shared__ float blockB[KNN_BLOCK_SIZE][KNN_BLOCK_SIZE];

// A is responsible for points indexed between aStart and aEnd
auto aStart = blockIdx.x * blockDim.x;
// B is responsible for points indexed between bStart and bEnd
auto bStart = blockIdx.y * blockDim.y;

auto ax = aStart + threadIdx.x;
auto bx = bStart + threadIdx.y;

auto sqDist = 0.0f;

auto numBlocksVertical = (d - 1) / KNN_BLOCK_SIZE + 1;

// Number of blocks that can be stored along the vertical dimension = gridDim.y
// Therefore this loop runs for each block along the vertical dimension
for(auto i = 0; i < numBlocksVertical; ++i)
{
// The i'th block on the vertical
auto startY = i * KNN_BLOCK_SIZE;
auto currY  = startY + threadIdx.y;

// The first part of the algorithm has each thread responsible
// for loading the values into blockA and blockB
if(startY + threadIdx.y < d)
{
if(ax < n)
blockA[threadIdx.y][threadIdx.x] = dIn[ax * d + currY];
if(bx < n)
blockB[threadIdx.y][threadIdx.x] = dIn[(bStart + threadIdx.x) * d + currY];
}

__syncthreads();

// Since <a,a> = a1 * a1 + a2 * a2 + a3 * a3 + ... + ad * ad
// We can compute the partial sum a1 * a1 + a2 * a2 + a3 * a3 + ... + ak * ak s.t k < d
// Each thread is now responsible for computing the partial sum of their respective element
// If the respective element is out of bounds, this loop can be skipped
if(ax < n && bx < n)
for(auto j = 0; j < KNN_BLOCK_SIZE; ++j)
{
auto diff = blockA[j][threadIdx.x] - blockB[j][threadIdx.y];
sqDist += diff * diff;
}
}

if(ax < n && bx < n)
{
dOut[ax * n + bx] = ax == bx ? INFINITY : sqDist;
}
}