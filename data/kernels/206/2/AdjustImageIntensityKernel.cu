#include "includes.h"

#define BLOCK_SIZE 512
#define BLOCK_SIZE_HOUGH 360
#define STEP_SIZE 5
#define NUMBER_OF_STEPS 360/STEP_SIZE

// Circ mask kernel storage
__constant__ int maskKernelX[NUMBER_OF_STEPS];
__constant__ int maskKernelY[NUMBER_OF_STEPS];

// Function to set precalculated relative coordinates for circle boundary coordinates
__global__ void AdjustImageIntensityKernel(float *imgOut, float *imgIn, int width, int height, float lowin, float lowout, float scale)
{
__shared__ float bufData[BLOCK_SIZE];

// Get the index of pixel
const int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;

// Load data to shared variable
bufData[threadIdx.x] = imgIn[index];

// Check that it's not out of bounds
if (index < (height*width)) {

// Find the according multiplier
float tempLevel = ( bufData[threadIdx.x] - lowin)*scale + lowout;

// Check that it's within required range
if (tempLevel < 0) {
bufData[threadIdx.x] = 0;
}
else if (tempLevel > 1) {
bufData[threadIdx.x] = 1;
}
else {
bufData[threadIdx.x] = tempLevel;
}

// Write data back
imgOut[index] = bufData[threadIdx.x];
}

// Synchronise threads to have the whole image fully processed for output
__syncthreads();
}