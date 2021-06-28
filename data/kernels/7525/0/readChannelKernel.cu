#include "includes.h"


#define BLOCK_SIZE 16
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16

// STD includes

// CUDA runtime

// Utilities and system includes


static // Print device properties
__global__ void readChannelKernel(unsigned char * image, unsigned char *channel, int imageW, int imageH, int channelToExtract, int numChannels) {
int y = blockIdx.y * blockDim.y + threadIdx.y;
int x = blockIdx.x * blockDim.x + threadIdx.x;

int posIn = y  * (imageW*numChannels) + (x*numChannels) + channelToExtract;
int posOut = y * imageW + x;

channel[posOut] = image[posIn];


}