#include "includes.h"



#define PI 3.1415926536f

texture<float, 2, cudaReadModeElementType> texRef;
/*
* Paint a 2D texture with a moving red/green hatch pattern on a
* strobing blue background.  Note that this kernel reads to and
* writes from the texture, hence why this texture was not mapped
* as WriteDiscard.
*/

//=================================
// write to texture;
//=================================
enum colors
{
RED, GREEN, BLUE, ALPHA
};


__global__ void cuke_gradient(unsigned char *surface, int width, int height, size_t pitch, float t)
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

// in the case where, due to quantization into grids, we have
// more threads than pixels, skip the threads which don't
// correspond to valid pixels
if (x >= width || y >= height) return;

// get a pointer to the pixel at (x,y)
float* pixel = (float *)(surface + y*pitch) + 4*x;

pixel[RED]		= x/640.0f;
pixel[GREEN]	= y/480.0f;
pixel[BLUE]		= 0.0f;
pixel[ALPHA]	= 1.0f;
}