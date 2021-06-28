#include "includes.h"
__global__ void add( float *x, float *y, float *z, float *deltaX, float *deltaY, float *deltaZ ) {
int tid = blockIdx.x;    // this thread handles the data at its thread id
if (tid < N)
x[tid] = x[tid] + deltaX[tid];
if (tid < N)
y[tid] = y[tid] + deltaY[tid];
if (tid<N)
z[tid] = z[tid] + deltaZ[tid];

}