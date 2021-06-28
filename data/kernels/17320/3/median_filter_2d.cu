#include "includes.h"
__device__ int index(int x, int y, int width)
{
return (y * width) + x;
}
__device__ const int FILTER_SIZE = 9; __device__ const int FILTER_HALFSIZE = FILTER_SIZE >> 1;  __device__ void sort_bubble(float *x, int n_size)
{
for (int i = 0; i < n_size - 1; i++)
{
for(int j = 0; j < n_size - i - 1; j++)
{
if (x[j] > x[j+1])
{
float temp = x[j];
x[j] = x[j+1];
x[j+1] = temp;
}
}
}
}
__global__ void median_filter_2d(unsigned char* input, unsigned char* output, int width, int height)
{
const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

if((x<width) && (y<height))
{
const int color_tid = index(x,y,width);
float windowMedian[MAX_WINDOW*MAX_WINDOW];
int windowElements = 0;
for (int x_iter = x - FILTER_HALFSIZE; x_iter <= x + FILTER_HALFSIZE; x_iter ++)
{
for (int y_iter = y - FILTER_HALFSIZE; y_iter <= y + FILTER_HALFSIZE; y_iter++)
{
if (0<=x_iter && x_iter < width && 0 <= y_iter && y_iter < height)
{
windowMedian[windowElements++] = input[index(x_iter,y_iter,width)];
}
}
}
sort_bubble(windowMedian,windowElements);
output[color_tid] = windowMedian[windowElements/2];
}
}