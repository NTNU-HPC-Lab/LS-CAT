#include "includes.h"

#define BLOCK_SIZE 512
#define BLOCK_SIZE_HOUGH 360
#define STEP_SIZE 5
#define NUMBER_OF_STEPS 360/STEP_SIZE

// Circ mask kernel storage
__constant__ int maskKernelX[NUMBER_OF_STEPS];
__constant__ int maskKernelY[NUMBER_OF_STEPS];

// Function to set precalculated relative coordinates for circle boundary coordinates
__global__ void ImageScalingKernel(float *imgOut, float *imgIn, int width, int height)
{
__shared__ float inData[BLOCK_SIZE];
// Get the index of pixel
const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

// Load data to shared variable
inData[threadIdx.x] = imgIn[index];

if ( index < (width*height) ) {
imgOut[index] = inData[threadIdx.x] / (float)255;
}

__syncthreads();
}