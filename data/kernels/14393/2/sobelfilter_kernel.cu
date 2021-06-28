#include "includes.h"



// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
__global__ void sobelfilter_kernel(int iw, int ih, unsigned char *source, unsigned char *dest)
{
// Calculate our pixel's location
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;

// Operate only if we are in the correct boundaries
if(x > 0 && x < iw - 1 && y > 0 && y < ih - 1)
{
int gx = -source[iw*(y-1)+(x-1)] + source[iw*(y-1)+(x+1)] +
-2*source[iw*(y)+(x-1)] + 2*source[iw*(y)+(x+1)] +
-source[iw*(y+1)+(x-1)] + source[iw*(y+1)+(x+1)];
int gy = -source[iw*(y-1)+(x-1)] - 2*source[iw*(y-1)+(x)]
-source[iw*(y-1)+(x+1)] +
source[iw*(y+1)+(x-1)] + 2*source[iw*(y+1)+(x)] +
source[iw*(y+1)+(x+1)];
dest[iw*y+x] = (int) sqrt((float)(gx)*(float)(gx) + (float)(gy)*(float)(gy));
}
}