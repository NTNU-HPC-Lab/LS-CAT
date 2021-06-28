#include "includes.h"



// Possible weight coefficients for tracking cost evaluation :
// Gaussian discretisation
/*
*       1  4  6  4  1
*       4 16 24 16  4
*       6 24 36 24  6
*       4 16 24 16  4
*       1  4  6  4  1
*/


// Compute spatial derivatives using Scharr operator - Naive implementation..

// Compute spatial derivatives using Scharr operator - Naive implementation..

// Compute spatial derivatives using Sobel operator - Naive implementation..

// Compute spatial derivatives using Sobel operator - Naive implementation..



// Low pass gaussian-like filtering before subsampling

// Low pass gaussian-like filtering before subsampling

/*
// Upsample a picture using the "magic" kernel
*/
__global__ void kernelSmoothY(float const * in, int w, int h, float * out)
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

if(x >= w || y >= h)
return;

int a = y-2;
int b = y-1;
int c = y;
int d = y+1;
int e = y+2;

if(a < 0) a = 0;
if(b < 0) b = 0;
if(d >= h) d = h-1;
if(e >= h) e = h-1;

out[y*w+x] = 0.0625f*in[a*w+x] + 0.25f*in[b*w+x] + 0.375f*in[c*w+x] + 0.25f*in[d*w+x] + 0.0625f*in[e*w+x];
}