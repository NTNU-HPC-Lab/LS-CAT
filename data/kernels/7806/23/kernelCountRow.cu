#include "includes.h"
__global__ void kernelCountRow(int *voronoiPtr, short2 *patternPtr, int *count, int width, int min, int max, int *cboundary) {
// Get the row we are working on
int x = blockIdx.x * blockDim.x + threadIdx.x;

// Collect the boundary (up, left, down, right)
if (x > 0 && x <= max) {
cboundary[width * 0 + x] = voronoiPtr[min * width + x];
cboundary[width * 1 + x] = voronoiPtr[x * width + min];
cboundary[width * 2 + x] = voronoiPtr[max * width + x];
cboundary[width * 3 + x] = voronoiPtr[x * width + max];
}

// Actual counting
if (x < min || x >= max)
return ;

int xwidth = x * width;
int result = 0;
short2 t = patternPtr[xwidth + min];

// Keep jumping and counting
while (t.y > 0 && t.y < max) {
result += 1 + (t.x >> 2);
t = patternPtr[xwidth + t.y + 1];
}

count[x] = result;
}