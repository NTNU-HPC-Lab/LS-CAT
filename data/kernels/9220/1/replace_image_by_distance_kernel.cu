#include "includes.h"


// fill an image with a chekcer_board (BGR)
__global__ void replace_image_by_distance_kernel(const unsigned char *pImage, const float* pDepth, const unsigned char *pBackground, unsigned char *result, const float max_value, const unsigned int width, const unsigned int height, const unsigned int image_channels)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if (y >= height || x >= width)
return;

// get the depth of the current pixel
float z_distance = pDepth[y * width + x];

// replace part of the view
int index = (y * width + x) * 3;
if (isfinite(z_distance) && (z_distance > max_value))
{
result[index] = pBackground[index];
result[index + 1] = pBackground[index + 1];
result[index + 2] = pBackground[index + 2];
}
else
{
if (image_channels == 1)//gray image
{
int img_index = y * width + x;
result[index] = pImage[img_index];
result[index + 1] = pImage[img_index];
result[index + 2] = pImage[img_index];
}
else//color image
{
int img_index = (y * width + x) * image_channels;
result[index] = pImage[img_index];
result[index + 1] = pImage[img_index + 1];
result[index + 2] = pImage[img_index + 2];
}
}
}