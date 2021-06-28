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




__device__ int CountAliveCells(int *i_cells, int idx, int width, int height)
{
int alive = 0;

int posY = floorf(idx / width);
int posX = idx % width;

for (int i = -1; i <= 1; i++)
{
for (int j = -1; j <= 1; j++)
{
int currPosX = (posX + i) % width;
int currPosY = (posY + j) % height;

if (currPosX < 0)
{
currPosX = width + currPosX;
}
if (currPosY < 0)
{
currPosY = height + currPosY;
}

int neigh = currPosY * width + currPosX;

if (i == 0 && j == 0)
continue;
if (i_cells[neigh] == 1)
alive++;
}
}
return alive;
}
__global__ void CalcNextGeneration(int *i_cells, int *o_cells, int width, int height)
{
int idx = blockDim.x * blockIdx.x + threadIdx.x;

if (idx >= width*height)
return;

int neighCount = CountAliveCells(i_cells, idx, width, height);


if (neighCount == 3 ||
(neighCount == 2 && i_cells[idx] == 1))
o_cells[idx] = 1;
else
o_cells[idx] = 0;

__syncthreads();

}