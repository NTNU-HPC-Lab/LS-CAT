#include "includes.h"
__global__ void sobelFilterShared3(unsigned char* g_DataIn, unsigned char * g_DataOut, unsigned int width, unsigned int height){
__shared__ char sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

int x = blockIdx.x * TILE_WIDTH + threadIdx.x - FILTER_RADIUS;
int y = blockIdx.y * TILE_HEIGHT + threadIdx.y - FILTER_RADIUS;

//Clamp to the center
x = max(FILTER_RADIUS, x);
x = min(x, width - FILTER_RADIUS - 1);
y = max(FILTER_RADIUS, y);
y = min(y, height - FILTER_RADIUS - 1);

unsigned int index = y * width + x;
unsigned int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

sharedMem[sharedIndex] = g_DataIn[index];

__syncthreads();

if(		threadIdx.x >= FILTER_RADIUS && threadIdx.x < BLOCK_WIDTH - FILTER_RADIUS
&&	threadIdx.y >= FILTER_RADIUS && threadIdx.y < BLOCK_HEIGHT - FILTER_RADIUS)
{
int sum = 0;

for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
{
int pixelValue = (int)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);

sum += pixelValue;
}

g_DataOut[index] = (unsigned char)(sum / FILTER_AREA);
}
}