#include "includes.h"
__global__ void rotateCuda( const uint8_t * in, uint32_t rowSizeIn, uint8_t * out, uint32_t rowSizeOut, float inXStart, float inYStart, uint32_t width, uint32_t height, float cosAngle, float sinAngle )
{
uint32_t outX = blockDim.x * blockIdx.x + threadIdx.x;
uint32_t outY = blockDim.y * blockIdx.y + threadIdx.y;

// Only do something if this thread is for a valid pixel in the output
if ( outX < width && outY < height ) {
// Both input coordinates are shifted using the cosAngle, sinAngle, outX, and outY. The shift
// comes from inverse rotating the horizontal and vertical iterations over the output.

// Note that inverse rotation by X axis is [cos(angle), -sin(angle)],
//   and the inverse rotation by Y axis is [sin(angle),  cos(angle)].

const float exactInX = inXStart + cosAngle * outX + sinAngle * outY;
const float exactInY = inYStart - sinAngle * outX + cosAngle * outY;

const int32_t inX = static_cast<int32_t>(exactInX);
const int32_t inY = static_cast<int32_t>(exactInY);

// Shift to the output pixel
out = out + outY * rowSizeOut + outX;

// Note that we will be taking an average with next pixels, so next pixels need to be in the image too
if ( inX < 0 || inX >= width - 1 || inY < 0 || inY >= height - 1 ) {
*out = 0; // We do not actually know what is beyond the image, so set value to 0
}
else {
// Shift to the input pixel
in = in + inY * rowSizeIn + inX;

// Now we use a bilinear approximation to find the pixel intensity value. That is, we take an
// average of pixels (inX, inY), (inX + 1, inY), (inX, inY + 1), and (inX + 1, inY + 1).
// We add an offset of 0.5 so that conversion to integer is done using rounding.
const float probX = exactInX - inX;
const float probY = exactInY - inY;
const float mean = *in * (1 - probX) * (1 - probY) +
*(in + 1) * probX * (1 - probY) +
*(in + rowSizeIn) * (1 - probX) * probY +
*(in + rowSizeIn + 1) * probX * probY +
0.5f;

*out = static_cast<uint8_t>(mean);
}
}
}