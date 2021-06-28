#include "includes.h"
__global__ void grayscale(unsigned char *src, unsigned char *dest, int width, int height, int nChannels) {
int y = blockIdx.y * blockDim.y + threadIdx.y;
int x = blockIdx.x * blockDim.x + threadIdx.x;

if(y < height && x < width) {
int pos = (y * width + x) * nChannels;

float r = src[pos + 2];
float g = src[pos + 1];
float b = src[pos + 0];

dest[pos + 2] = ((0.393f * r + 0.769f * g + 0.189f * b) > 255) ? 255 : (0.393f * r + 0.769f * g + 0.189f * b);
dest[pos + 1] = ((0.349f * r + 0.686f * g + 0.168f * b) > 255) ? 255 : (0.349f * r + 0.686f * g + 0.168f * b);
dest[pos + 0] = ((0.272f * r + 0.534f * g + 0.131f * b) > 255) ? 255 : (0.272f * r + 0.534f * g + 0.131f * b);
}
}