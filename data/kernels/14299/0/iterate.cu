#include "includes.h"

#define THREADSPBLK 1024
#define THREADSPSM 2048
#define TILE_WIDTH 32
#define TOTAL_ITERATIONS 50

int main_n;


__global__ void iterate(float* originalMatrixD, float* solutionD, int originalMatrixWidth, int startingIndex) {
// __shared__ float originalMatrixDS [TILE_WIDTH][TILE_WIDTH];
__shared__ float originalMatrixDS [TILE_WIDTH * TILE_WIDTH];

int tx = threadIdx.x;
int ty = threadIdx.y;

int blockId = blockIdx.x + blockIdx.y * gridDim.x;

int currentMatrixIndex = blockId * (blockDim.x * blockDim.y) +
(threadIdx.y * blockDim.x) + threadIdx.x;

currentMatrixIndex += startingIndex;

originalMatrixDS[ty * TILE_WIDTH + tx] = originalMatrixD[currentMatrixIndex];

// Sync up w/ shared data set up
__syncthreads();

float replaceAmount;
bool onEdge = false;
int XEdgeCheckMod = currentMatrixIndex % originalMatrixWidth;

// X = 0 edge
if ( XEdgeCheckMod == 0) {
onEdge = true;
}

// X = N - 1
else if ( XEdgeCheckMod == (originalMatrixWidth - 1)) {
onEdge = true;
}

// Y = 0
else if (currentMatrixIndex < originalMatrixWidth) {
onEdge = true;
}

// Y = N - 1
else if (currentMatrixIndex >= (originalMatrixWidth * originalMatrixWidth
- originalMatrixWidth)) {
onEdge = true;
}

if (onEdge) {
replaceAmount = originalMatrixDS[ty * TILE_WIDTH + tx];
}

else {
// Top and Bottom come from Global memory
float top = originalMatrixD[currentMatrixIndex - originalMatrixWidth];
float bottom = originalMatrixD[currentMatrixIndex + originalMatrixWidth];
float left;
float right;

// Left and right edge come from Global memory
if (tx == 0 && ty == 0) {
left = originalMatrixD[currentMatrixIndex - 1];
}

else {
left = originalMatrixDS[ty * TILE_WIDTH + tx - 1];
}

if ((ty == TILE_WIDTH - 1) && (tx == TILE_WIDTH - 1)) {
right = originalMatrixD[currentMatrixIndex + 1];
}

else {
right = originalMatrixDS[ty * TILE_WIDTH + tx + 1];
}

replaceAmount = (left + right + top + bottom) / 4;
}

solutionD[currentMatrixIndex] = replaceAmount;
}