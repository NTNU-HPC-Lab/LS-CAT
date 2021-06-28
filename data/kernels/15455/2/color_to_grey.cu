#include "includes.h"
__global__ void color_to_grey(uchar3 *input_image, uchar3 *output_image, int width, int height)
{
int col = threadIdx.x + blockIdx.x * blockDim.x;
int row = threadIdx.y + blockIdx.y * blockDim.y;

if(col < width && row < height)
{
int pos = row * width + col;
output_image[pos].x = static_cast<unsigned char>(input_image[pos].x * 0.2126f + input_image[pos].y * 0.7125f + input_image[pos].z * 0.0722f);
output_image[pos].y = static_cast<unsigned char>(input_image[pos].x * 0.2126f + input_image[pos].y * 0.7125f + input_image[pos].z * 0.0722f);
output_image[pos].z = static_cast<unsigned char>(input_image[pos].x * 0.2126f + input_image[pos].y * 0.7125f + input_image[pos].z * 0.0722f);
}
}