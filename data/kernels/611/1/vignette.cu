#include "includes.h"
__global__ void vignette(const unsigned char * src, unsigned char * dst, float inner, float outer, const size_t width, const size_t height)
{
// the xIndex and yIndex will be used cordinates pixels of the image
// NOTE
// NOTE This assumes that we are treating this as a two dimensional data structure and the blocks will be used in the same way
// NOTE
size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;

// Checking to see if the indexs are within the bounds of the image
if (xIndex < width && yIndex < height)
{
// offset represents postion of the current pixel in the one dimensional array
size_t offset = yIndex * width + xIndex;
// Shift the pixel oriented coordinates into image resolution independent coordinates
// where 0, 0 is the center of the image.
float x = xIndex / float(height) - float(width) / float(height) / 2.0f;
float y = yIndex / float(height) - 0.5f;
//Calculates current pixels distance from the center where the cordinates are 0, 0
float d = sqrtf(x * x + y * y);
if (d < inner)
{
// if d is less than inner boundary, we don't change that specific image pixel
*(dst + offset) = *(src + offset);
}
else if (d > outer)
{
// if d is greater than outer boundary, we set it to 0 so it becomes black
*(dst + offset) = 0;
}
else
{
// If in between the inner and outer boundaries, it will be a shade of gray.
// NOTE
// NOTE  This assumes... by the time we get here, we have checked that outer does not equal inner
// NOTE  This also assumes ... by the time we get here, we have made inner less than outer
// NOTE
float v = 1 - (d - inner) / (outer - inner);
*(dst + offset) = (unsigned char)(*(src + offset) * v);
}
}
}