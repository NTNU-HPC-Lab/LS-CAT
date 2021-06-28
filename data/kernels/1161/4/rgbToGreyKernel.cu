#include "includes.h"
__global__ void rgbToGreyKernel(int height,int width ,unsigned char *input_img, unsigned char *output_img)
{
int col = blockIdx.x*blockDim.x + threadIdx.x;
int row = blockIdx.y*blockDim.y + threadIdx.y;

if(row<height && col<width)
{
int idx = row*width + col;
float red = (float)input_img[3*idx];
float green = (float)input_img[3*idx+1];
float blue = (float)input_img[3*idx+2];

output_img[idx] = 0.21*red + 0.71*green + 0.07*blue;
}

}