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
__global__ void kernelMagicUpsampleX(float *in, int _w, int _h, float *out) {
// Coefficients : 1/4, 3/4, 3/4, 1/4 in each direction (doubles the size of the picture)

int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

if(x >= _w || y >= _h)
return;

// Duplicate the points at the same place (?)
out[y*2*_w + 2*x] = in[y*_w+x];


if ((x < (_w-2)) && (x > 1))
out[y*2*_w + 2*x + 1] = __fdividef(3.0*(in[y*_w+x] + in[y*_w + x + 1]) + in[y*_w+x -1] + in[y*_w+x +2] , 8.0);

}