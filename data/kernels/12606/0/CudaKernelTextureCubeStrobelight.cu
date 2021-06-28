#include "includes.h"
/* ==========================================================================
textureCube.cu
==========================================================================

Main wrapper + kernel that changes the colors of the four faces

*/




#define PI 3.1415926536f

// --------------------------------------------------------------------------
// Kernel
// --------------------------------------------------------------------------


// Paint a 2D surface with a moving bulls-eye pattern.  The "face" parameter selects
// between 6 different colors to use.  We will use a different color on each face of a
// cube map.


// --------------------------------------------------------------------------
// Wrapper
// --------------------------------------------------------------------------

// Sets up grid / blocks, launches kernel
extern "C"
__global__ void CudaKernelTextureCubeStrobelight(char *surface, int width, int height, size_t pitch, int face, float t)
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;
unsigned char *pixel;

// in the case where, due to quantization into grids, we have
// more threads than pixels, skip the threads which don't
// correspond to valid pixels
if (x >= width || y >= height) return;

// get a pointer to this pixel
pixel = (unsigned char *)(surface + y*pitch) + 4 * x;

// populate it
float theta_x = (2.0f*x) / width - 1.0f;
float theta_y = (2.0f*y) / height - 1.0f;
float theta = 2.0f*PI*sqrt(theta_x*theta_x + theta_y*theta_y);
unsigned char value = 255 * (0.6f + 0.4f*cos(theta + t));

pixel[3] = 255; // alpha

if (face % 2)
{
pixel[0] =    // blue
pixel[1] =    // green
pixel[2] = 0.5; // red
pixel[face / 2] = value;
}
else
{
pixel[0] =        // blue
pixel[1] =        // green
pixel[2] = value; // red
pixel[face / 2] = 0.5;
}
}