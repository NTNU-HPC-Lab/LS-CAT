#include "includes.h"
__global__ void gpu_grayscale(int width, int height, float *image, float *image_out)
{
////////////////
// TO-DO #4.2 /////////////////////////////////////////////
// Implement the GPU version of the grayscale conversion //
///////////////////////////////////////////////////////////

const int x = blockIdx.x*blockDim.x + threadIdx.x;
const int y = blockIdx.y*blockDim.y + threadIdx.y;

if (x < width && y < height) {

int offset_out = ((width * y) + x);
int offset = offset_out*3;

float *pixel = &image[offset];

image_out[offset_out] = pixel[0] * 0.0722f + // B
pixel[1] * 0.7152f + // G
pixel[2] * 0.2126f;  // R

}
}