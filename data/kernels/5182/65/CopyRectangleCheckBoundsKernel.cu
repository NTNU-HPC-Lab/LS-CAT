#include "includes.h"
__global__ void CopyRectangleCheckBoundsKernel(	float *src, int srcOffset, int srcWidth, int srcHeight, int srcRectX, int srcRectY, int rectWidth, int rectHeight, float *dest, int destOffset, int destWidth, int destRectX, int destRectY, float defaultValue )
{
int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

int size = rectWidth * rectHeight;

if (id < size) {

int localX = id % rectWidth;
int localY = id / rectWidth;

int srcPixelX = srcRectX + localX;
int srcPixelY = srcRectY + localY;

int destPixelX = destRectX + localX;
int destPixelY = destRectY + localY;

if (srcPixelX >= 0 && srcPixelX < srcWidth && srcPixelY >= 0 && srcPixelY < srcHeight)
{
(dest + destOffset)[destPixelX + destPixelY * destWidth] = (src + srcOffset)[srcPixelX + srcPixelY * srcWidth];
}
else
{
(dest + destOffset)[destPixelX + destPixelY * destWidth] = defaultValue;
}
}
}