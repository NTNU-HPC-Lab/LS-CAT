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
__global__ void kernelSobelY( float const *in, int _w, int _h, float *out)
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

if(x >= _w || y >= _h)
return;

// Pattern  // Indexes:
//  -1 0 1 // a1 b1 c1
// - 2 0 2 // a2 b2 c2
//  -1 0 1 // a3 b3 c3

int a = max(y-1,0);
int c = min((y+1),_h -1);

int a1, a2, a3,
c1, c2, c3;

int i1 = max(x-1, 0);
int i3 = min(x+1, _w-1);

a1 = a*_w + i1;
a2 = a*_w + x;
a3 = a*_w + i3;

c1 = c*_w + i1;
c2 = c*_w + x;
c3 = c*_w + i3;

out[y*_w + x] = __fdividef(-1.0*in[a1] -2.0*in[a2] -1.0*in[a3] + 1.0*in[c1] + 2.0*in[c2] + 1.0*in[c3], 4.0);
}