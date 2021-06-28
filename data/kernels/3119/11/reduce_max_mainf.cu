#include "includes.h"
/// ================================================================
///
/// Disclaimer:  IMPORTANT:  This software was developed at theNT
/// National Institute of Standards and Technology by employees of the
/// Federal Government in the course of their official duties.
/// Pursuant to title 17 Section 105 of the United States Code this
/// software is not subject to copyright protection and is in the
/// public domain.  This is an experimental system.  NIST assumes no
/// responsibility whatsoever for its use by other parties, and makes
/// no guarantees, expressed or implied, about its quality,
/// reliability, or any other characteristic.  We would appreciate
/// acknowledgement if the software is used.  This software can be
/// redistributed and/or modified freely provided that any derivative
/// works bear some notice that they are derived from it, and any
/// modified versions bear some notice that they have been modified.
///
/// ================================================================

// ================================================================
//
// Author: Timothy Blattner
// Date:   Wed Nov 30 12:36:40 2011 EScufftDoubleComplex
//
// Functions that execute on the graphics card for doing
// Vector computation.
//
// ================================================================


#define THREADS_PER_BLOCK 256
#define MIN_DISTANCE 1.0

// ================================================================
__global__ void reduce_max_mainf(float *g_idata, float *g_odata, int * max_idx, unsigned int n, int blockSize)
{
__shared__ float sdata[THREADS_PER_BLOCK];
__shared__ int idxData[THREADS_PER_BLOCK];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockSize) + tid;
unsigned int gridSize = blockSize*gridDim.x;


float myMax = 0.0;
int myMaxIndex;
float val;

while (i < n)
{
val = g_idata[i];
if (myMax < val)
{
myMax = val;
myMaxIndex = i;
}

if (i+blockSize < n)
{
val = g_idata[i+blockSize];
if (myMax < val)
{
myMax = val;
myMaxIndex = i+blockSize;
}
}

i += gridSize;
}

sdata[tid] = myMax;
idxData[tid] = myMaxIndex;

__syncthreads();

if (blockSize >= 512)
{
if (tid < 256)
{
if (myMax < sdata[tid + 256])
{
sdata[tid] = myMax = sdata[tid+256];
idxData[tid] = idxData[tid+256];
}
}
__syncthreads();
}

if (blockSize >= 256)
{
if (tid < 128)
{
if (myMax < sdata[tid + 128])
{
sdata[tid] = myMax = sdata[tid+128];
idxData[tid] = idxData[tid+128];
}
}
__syncthreads();
}

if (blockSize >= 128)
{
if (tid <   64)
{
if(myMax < sdata[tid +   64])
{
sdata[tid] = myMax = sdata[tid+64];
idxData[tid] = idxData[tid+64];
}
}
__syncthreads();
}

volatile float *vdata = sdata;
volatile int *vidxData = idxData;

if (tid < 32)
{
if (blockSize >=  64)
if (myMax < vdata[tid + 32])
{
vdata[tid] = myMax = vdata[tid+32];
vidxData[tid] = vidxData[tid+32];
}

if (blockSize >=  32)
if (myMax < vdata[tid + 16])
{
vdata[tid] = myMax = vdata[tid+16];
vidxData[tid] = vidxData[tid+16];
}

if (blockSize >=  16)
if (myMax < vdata[tid +  8])
{
vdata[tid] = myMax = vdata[tid+8];
vidxData[tid] = vidxData[tid+8];
}

if (blockSize >=    8)
if (myMax < vdata[tid +  4])
{
vdata[tid] = myMax = vdata[tid+4];
vidxData[tid] = vidxData[tid+4];
}

if (blockSize >=    4)
if (myMax < vdata[tid+2])
{
vdata[tid] = myMax = vdata[tid+2];
vidxData[tid] = vidxData[tid+2];
}

if (blockSize >=    2)
if (myMax < vdata[tid +  1])
{
vdata[tid] = myMax = vdata[tid+1];
vidxData[tid] = vidxData[tid+1];
}
__syncthreads();
}

if (tid == 0)
{
g_odata[blockIdx.x] = sdata[0];
max_idx[blockIdx.x] = idxData[0];
}
}