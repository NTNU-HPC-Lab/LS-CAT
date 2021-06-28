#include "includes.h"


#define BLOCK_SIZE 16
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

// STD includes

// CUDA runtime

// Utilities and system includes


static // Print device properties
__global__ void writeChannelKernel( unsigned char* image, unsigned char* channel, int imageW, int imageH, int channelToMerge, int numChannels) {
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int posOut = y * (imageW*numChannels) + (x*numChannels) + channelToMerge;
int posIn = y * imageW + x;

image[posOut] = channel[posIn];

}