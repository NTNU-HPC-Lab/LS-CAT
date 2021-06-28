#include "includes.h"
/*******************************************************************************
*
*******************************************************************************/

/*************************************************************************

/*************************************************************************/

/*************************************************************************/
__global__ void drawGray(unsigned char* optr, const float* outSrc) {
// map from threadIdx/BlockIdx to pixel position
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x;

float val = outSrc[offset];

val = (val / 50.0) + 0.5; //get {-25 to 25} range into {0 to 1} range
if (val < 0) val = 0;
if (val > 1) val = 1;

optr[offset * 4 + 0] = 255 * val;       // red
optr[offset * 4 + 1] = 255 * val;       // green
optr[offset * 4 + 2] = 255 * val;       // blue
optr[offset * 4 + 3] = 255;             // alpha (opacity)
}