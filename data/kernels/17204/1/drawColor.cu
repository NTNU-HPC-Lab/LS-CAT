#include "includes.h"
/*******************************************************************************
*
*******************************************************************************/

/*************************************************************************

/*************************************************************************/

/*************************************************************************/
__global__ void drawColor(unsigned char* optr, const float* red, const float* green, const float* blue) {
// map from threadIdx/BlockIdx to pixel position
int x = threadIdx.x + blockIdx.x * blockDim.x;
int y = threadIdx.y + blockIdx.y * blockDim.y;
int offset = x + y * blockDim.x * gridDim.x;

float theRed = red[offset];
//  theRed = (theRed / 50.0) + 0.5;
if (theRed < 0) theRed = 0;
if (theRed > 1) theRed = 1;

float theGreen = green[offset];
//  theGreen = (theGreen / 50.0) + 0.5;
if (theGreen < 0) theGreen = 0;
if (theGreen > 1) theGreen = 1;

float theBlue = blue[offset];
//  theBlue = (theBlue / 50.0) + 0.5;
if (theBlue < 0) theBlue = 0;
if (theBlue > 1) theBlue = 1;


optr[offset * 4 + 0] = 255 * theRed;    // red
optr[offset * 4 + 1] = 255 * theGreen;  // green
optr[offset * 4 + 2] = 255 * theBlue;   // blue
optr[offset * 4 + 3] = 255;             // alpha (opacity)
}