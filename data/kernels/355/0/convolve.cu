#include "includes.h"
//
//  imgproc_main.cpp
//
//
//  Created by Nathaniel Lewis on 3/8/12.
//  Copyright (c) 2012 E1FTW Games. All rights reserved.
//



// GPU constant memory to hold our kernels (extremely fast access time)
__constant__ float convolutionKernelStore[256];

/**
* Convolution function for cuda.  Destination is expected to have the same width/height as source, but there will be a border
* of floor(kWidth/2) pixels left and right and floor(kHeight/2) pixels top and bottom
*
* @param source      Source image host pinned memory pointer
* @param width       Source image width
* @param height      Source image height
* @param paddingX    source image padding along x
* @param paddingY    source image padding along y
* @param kOffset     offset into kernel store constant memory
* @param kWidth      kernel width
* @param kHeight     kernel height
* @param destination Destination image host pinned memory pointer
*/

// converts the pythagoran theorem along a vector on the GPU

// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
__global__ void convolve(unsigned char *source, int width, int height, int paddingX, int paddingY, ssize_t kOffset, int kWidth, int kHeight, unsigned char *destination)
{
// Calculate our pixel's location
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;

float sum = 0.0;
int   pWidth = kWidth/2;
int   pHeight = kHeight/2;

// Only execute for valid pixels
if(x >= pWidth+paddingX &&
y >= pHeight+paddingY &&
x < (blockDim.x * gridDim.x)-pWidth-paddingX &&
y < (blockDim.y * gridDim.y)-pHeight-paddingY)
{
for(int j = -pHeight; j <= pHeight; j++)
{
for(int i = -pWidth; i <= pWidth; i++)
{
// Sample the weight for this location
int ki = (i+pWidth);
int kj = (j+pHeight);
float w  = convolutionKernelStore[(kj * kWidth) + ki + kOffset];


sum += w * float(source[((y+j) * width) + (x+i)]);
}
}
}

// Average the sum
destination[(y * width) + x] = (unsigned char) sum;
}