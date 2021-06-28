#include "includes.h"



// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
__global__ void boxfilter_kernel(int iw, int ih, unsigned char *source, unsigned char *dest, int bw, int bh)
{
// Calculate our pixel's location
int x = (blockIdx.x * blockDim.x) + threadIdx.x;
int y = (blockIdx.y * blockDim.y) + threadIdx.y;

// Variables to store the sum
int count = 0;
float sum = 0.0;

// Do the blur operation by summing the surround pixels
for(int j = -(bh/2); j <= (bh/2); j++)
{
for(int i = -(bw/2); i <= (bw/2); i++)
{
// Verify that this offset is within the image boundaries
if((x+i) < iw && (x+i) >= 0 && (y+j) < ih && (y+j) >= 0)
{
sum += (float) source[((y+j) * iw) + (x+i)];
count++;
}
}
}

// Average the sum
sum /= (float) count;
dest[(y * iw) + x] = (unsigned char) sum;
}