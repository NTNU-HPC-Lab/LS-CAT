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
__global__ void pythagoras(unsigned char *a, unsigned char *b, unsigned char *c)
{
int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

float af = float(a[idx]);
float bf = float(b[idx]);

c[idx] = (unsigned char) sqrtf(af*af + bf*bf);
}