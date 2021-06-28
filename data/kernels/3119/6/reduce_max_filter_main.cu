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
__device__ bool checkDistance(volatile int *maxesRow, volatile int *maxesCol, int nMax, int curIdx, int width)
{
int row = curIdx / width;
int col = curIdx % width;
int j;
//double dist;
for (j = 0; j < nMax; j++)
{

if (maxesRow[j] == row && maxesCol[j] == col)
return false;

//		dist = distance(maxesRow[j], row, maxesCol[j], col);

//		if (dist < MIN_DISTANCE)
//			return false;


}

return true;
}
__device__ bool checkDistance(int *maxesRow, int *maxesCol, int nMax, int curIdx, int width)
{
int row = curIdx / width;
int col = curIdx % width;
int j;
//double dist;
for (j = 0; j < nMax; j++)
{
if (maxesRow[j] == row && maxesCol[j] == col)
return false;

//dist = distance(maxesRow[j], row, maxesCol[j], col);

//if (dist < MIN_DISTANCE)
//	return false;


}

return true;
}
__device__ double distance(int x1, int x2, int y1, int y2)
{
return ((double(x1-x2))*(double(x1-x2)))+
((double(y1-y2))*(double(y1-y2)));
}
__global__ void reduce_max_filter_main(double *g_idata, double *g_odata, int * max_idx, unsigned int width, unsigned int height, int blockSize, int *maxes, int nMax)
{
__shared__ int smaxesRow[10];
__shared__ int smaxesCol[10];
__shared__ int smaxesVal[10];
__shared__ double sdata[THREADS_PER_BLOCK];
__shared__ int idxData[THREADS_PER_BLOCK];
unsigned int tid = threadIdx.x;
unsigned int i = blockIdx.x*(blockSize) + tid;
unsigned int gridSize = blockSize*gridDim.x;
if (tid < nMax)
{
smaxesVal[tid] = maxes[tid];
smaxesRow[tid] = smaxesVal[tid] / width;
smaxesCol[tid] = smaxesVal[tid] % width;
}
__syncthreads();

double myMax = -INFINITY;
int myMaxIndex;
double val;

while (i < width * height)
{
val = g_idata[i];
if (myMax < val)
{
// compute distance . . .
if (checkDistance(smaxesRow, smaxesCol,
nMax, i, width))
{
myMax = val;
myMaxIndex = i;
}
}

if (i+blockSize < width * height)
{
val = g_idata[i+blockSize];
if (myMax < val)
{

if (checkDistance(smaxesRow, smaxesCol,
nMax, i+blockSize, width))
{
myMax = val;
myMaxIndex = i+blockSize;
}
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
if (checkDistance(smaxesRow, smaxesCol,
nMax, idxData[tid+256],
width))
{
sdata[tid] = myMax = sdata[tid+256];
idxData[tid] = idxData[tid+256];
}
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
if (checkDistance(smaxesRow, smaxesCol,
nMax, idxData[tid+128],
width))
{
sdata[tid] = myMax = sdata[tid+128];
idxData[tid] = idxData[tid+128];
}
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
if (checkDistance(smaxesRow, smaxesCol,
nMax, idxData[tid+64],
width))
{
sdata[tid] = myMax = sdata[tid+64];
idxData[tid] = idxData[tid+64];
}
}
}
__syncthreads();
}

volatile double *vdata = sdata;
volatile int *vidxData = idxData;

volatile int *vsmaxesRow = smaxesRow;
volatile int *vsmaxesCol = smaxesCol;

if (tid < 32)
{
if (blockSize >=  64)
if (myMax < vdata[tid + 32])
{
if (checkDistance(vsmaxesRow, vsmaxesCol,
nMax, vidxData[tid+32],
width))
{
vdata[tid] = myMax = vdata[tid+32];
vidxData[tid] = vidxData[tid+32];
}
}

if (blockSize >=  32)
if (myMax < vdata[tid + 16])
{

if (checkDistance(vsmaxesRow, vsmaxesCol,
nMax, vidxData[tid+16],
width))
{
vdata[tid] = myMax = vdata[tid+16];
vidxData[tid] = vidxData[tid+16];
}
}

if (blockSize >=  16)
if (myMax < vdata[tid +  8])
{
if (checkDistance(vsmaxesRow, vsmaxesCol,
nMax, vidxData[tid+8],
width))
{
vdata[tid] = myMax = vdata[tid+8];
vidxData[tid] = vidxData[tid+8];
}
}

if (blockSize >=    8)
if (myMax < vdata[tid +  4])
{
if (checkDistance(vsmaxesRow, vsmaxesCol,
nMax, vidxData[tid+4],
width))
{
vdata[tid] = myMax = vdata[tid+4];
vidxData[tid] = vidxData[tid+4];
}
}

if (blockSize >=    4)
if (myMax < vdata[tid+2])
{
if (checkDistance(vsmaxesRow, vsmaxesCol,
nMax, vidxData[tid+2],
width))
{
vdata[tid] = myMax = vdata[tid+2];
vidxData[tid] = vidxData[tid+2];
}
}

if (blockSize >=    2)
if (myMax < vdata[tid +  1])
{
if (checkDistance(vsmaxesRow, vsmaxesCol,
nMax, vidxData[tid+1],
width))
{
vdata[tid] = myMax = vdata[tid+1];
vidxData[tid] = vidxData[tid+1];
}
}
__syncthreads();
}

if (tid == 0)
{
g_odata[blockIdx.x] = sdata[0];
max_idx[blockIdx.x] = idxData[0];

if (gridDim.x == 1)
maxes[nMax] = idxData[0];
}
}