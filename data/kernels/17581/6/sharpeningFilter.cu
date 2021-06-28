#include "includes.h"
__global__ void sharpeningFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height, int channel)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

float kernel[FILTER_WIDTH][FILTER_HEIGHT] = { {-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1} };
// only threads inside image will write results
if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
{
for (int c = 0; c < channel; c++)
{
// Sum of pixel values
float sum = 0;
// Loop inside the filter to average pixel values
for (int ky = -FILTER_HEIGHT / 2; ky <= FILTER_HEIGHT / 2; ky++) {
for (int kx = -FILTER_WIDTH / 2; kx <= FILTER_WIDTH / 2; kx++) {
float fl = srcImage[((y + ky) * width + (x + kx)) * channel + c];
sum += fl * kernel[ky + FILTER_HEIGHT / 2][kx + FILTER_WIDTH / 2];
}
}
dstImage[(y * width + x) * channel + c] = sum;
}
}
}