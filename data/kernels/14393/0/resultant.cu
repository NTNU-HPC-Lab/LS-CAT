#include "includes.h"



// create an image buffer.  return host ptr, pass out device pointer through pointer to pointer
__global__ void resultant(unsigned char *a, unsigned char *b, unsigned char *c)
{
int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

float opposite_side = float(a[idx]);
float adjacent_side = float(b[idx]);

// Figure out the hypotenuse
c[idx] = (unsigned char) sqrtf((opposite_side + adjacent_side)*(opposite_side + adjacent_side ) - (2 * opposite_side * adjacent_side));
if ( c[idx] > 15 && c [idx -1] != 148 )
c[idx] = 148;
else
c[idx] = 0;

}