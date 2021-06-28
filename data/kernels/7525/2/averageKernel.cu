#include "includes.h"


#define BLOCK_SIZE 16
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

// STD includes

// CUDA runtime

// Utilities and system includes


static // Print device properties
__global__ void averageKernel( unsigned char* inputChannel, unsigned char* outputChannel, int imageW, int imageH)
{
int y = blockIdx.y * blockDim.y + threadIdx.y;
int x = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned int numElements = ((2 * KERNEL_RADIUS) + 1) * ((2 * KERNEL_RADIUS) + 1);

unsigned int sum = 0;
for (int kY = -KERNEL_RADIUS; kY <= KERNEL_RADIUS; kY++) {
const int curY = y + kY;
if (curY < 0 || curY > imageH) {
continue;
}

for (int kX = -KERNEL_RADIUS; kX <= KERNEL_RADIUS; kX++) {
const int curX = x + kX;
if (curX < 0 || curX > imageW) {
continue;
}

const int curPosition = (curY * imageW + curX);
if (curPosition >= 0 && curPosition < (imageW * imageH)) {
sum += inputChannel[curPosition];
}
}
}
outputChannel[y * imageW + x] = (unsigned char)(sum / numElements);
}