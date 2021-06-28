#include "includes.h"
////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
Max dimensions of the world: 6000 x 5500
Parameters: Intel Core i5-2500K 3.30GHz
8GB RAM
NVIDIA GeForce GTX 560 Ti
*/

// includes, system

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#endif

// includes, cuda

// Utilities and timing functions

// CUDA helper functions




__global__ void UpdateGrid(int offX, int offY ,int *i_cells, struct uchar4 *dst, int width, int height, int worldW, int worldH)
{
int idx = blockDim.x * blockIdx.x + threadIdx.x; //position in visible grid
int posX =offX + idx%width;
int posY =offY + idx / width;


int cellIdx = posX*worldW + posY; //calc position of cell in grid
if ( idx < width*height)
{
//assign color
dst[idx].x = i_cells[cellIdx] * 255;
dst[idx].y = i_cells[cellIdx] * 255;
dst[idx].z = i_cells[cellIdx] * 255;
}
}