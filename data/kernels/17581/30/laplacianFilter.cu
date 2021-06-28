#include "includes.h"
__global__ void laplacianFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height)
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

float ker[3][3] = {{0, -1, 0}, {-1, 4, -1}, {0, -1, 0}};
//float kernel[3][3] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
// only threads inside image will write results
if((x>=FILTER_WIDTH/2) && (x<(width-FILTER_WIDTH/2)) && (y>=FILTER_HEIGHT/2) && (y<(height-FILTER_HEIGHT/2)))
{
// Sum of pixel values
float sum = 0;
// Loop inside the filter to average pixel values
for(int ky=-FILTER_HEIGHT/2; ky<=FILTER_HEIGHT/2; ky++) {
for(int kx=-FILTER_WIDTH/2; kx<=FILTER_WIDTH/2; kx++) {
float fl = srcImage[((y+ky)*width + (x+kx))];
sum += fl*ker[ky+FILTER_HEIGHT/2][kx+FILTER_WIDTH/2];
}
}
dstImage[(y*width+x)] =  sum;
}
}