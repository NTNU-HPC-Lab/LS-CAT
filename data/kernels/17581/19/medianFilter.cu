#include "includes.h"
__device__ void sort(unsigned char* filterVector)
{
for (int i = 0; i < FILTER_WIDTH*FILTER_HEIGHT; i++) {
for (int j = i + 1; j < FILTER_WIDTH*FILTER_HEIGHT; j++) {
if (filterVector[i] > filterVector[j]) {
//Swap the variables
unsigned char tmp = filterVector[i];
filterVector[i] = filterVector[j];
filterVector[j] = tmp;
}
}
}
}
__global__ void medianFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height, int channel)
{
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;

// only threads inside image will write results
if((x>=FILTER_WIDTH/2) && (x<(width-FILTER_WIDTH/2)) && (y>=FILTER_HEIGHT/2) && (y<(height-FILTER_HEIGHT/2)))
{
for(int c=0 ; c<channel ; c++)
{
unsigned char filterVector[FILTER_WIDTH*FILTER_HEIGHT];
// Loop inside the filter to average pixel values
for(int ky=-FILTER_HEIGHT/2; ky<=FILTER_HEIGHT/2; ky++) {
for(int kx=-FILTER_WIDTH/2; kx<=FILTER_WIDTH/2; kx++) {
filterVector[ky*FILTER_WIDTH+kx] = srcImage[((y+ky)*width + (x+kx))*channel+c];
}
}
// Sorting values of filter
sort(filterVector);
dstImage[(y*width+x)*channel+c] =  filterVector[(FILTER_WIDTH*FILTER_HEIGHT)/2];
}
}
}