#include "includes.h"
__device__ int clamp(int value, int bound)
{
if (value < 0) {
return 1;
}
if (value < bound) {
return value;
}
return bound - 1;
}
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
__global__ void median_filter_2d_sm(unsigned char* input, unsigned char* output, int width, int height)
{
__shared__ int sharedPixels[BLOCKDIM + FILTER_SIZE][BLOCKDIM + FILTER_SIZE];

const int x = blockIdx.x * blockDim.x + threadIdx.x;
const int y = blockIdx.y * blockDim.y + threadIdx.y;

int xBlockLimit_max = blockDim.x - FILTER_HALFSIZE - 1;
int yBlockLimit_max = blockDim.y - FILTER_HALFSIZE - 1;
int xBlockLimit_min = FILTER_HALFSIZE;
int yBlockLimit_min = FILTER_HALFSIZE;

if (threadIdx.x > xBlockLimit_max && threadIdx.y > yBlockLimit_max) {
int i = index(clamp(x + FILTER_HALFSIZE,width), clamp(y + FILTER_HALFSIZE,height), width);
unsigned int pixel = input[i];
sharedPixels[threadIdx.x + 2*FILTER_HALFSIZE][threadIdx.y + 2*FILTER_HALFSIZE] = pixel;
}
if (threadIdx.x > xBlockLimit_max && threadIdx.y < yBlockLimit_min) {
int i = index(clamp(x + FILTER_HALFSIZE,width), clamp(y - FILTER_HALFSIZE,height), width);
unsigned int pixel = input[i];
sharedPixels[threadIdx.x + 2*FILTER_HALFSIZE][threadIdx.y] = pixel;
}
if (threadIdx.x < xBlockLimit_min && threadIdx.y > yBlockLimit_max) {
int i = index(clamp(x - FILTER_HALFSIZE,width), clamp(y + FILTER_HALFSIZE,height), width);
unsigned int pixel = input[i];
sharedPixels[threadIdx.x][threadIdx.y + 2*FILTER_HALFSIZE] = pixel;
}
if (threadIdx.x < xBlockLimit_min && threadIdx.y < yBlockLimit_min) {
int i = index(clamp(x - FILTER_HALFSIZE,width), clamp(y - FILTER_HALFSIZE,height), width);
unsigned int pixel = input[i];
sharedPixels[threadIdx.x][threadIdx.y] = pixel;
}
if (threadIdx.x < xBlockLimit_min) {
int i = index(clamp(x - FILTER_HALFSIZE,width), clamp(y,height), width);
unsigned int pixel = input[i];
sharedPixels[threadIdx.x][threadIdx.y + FILTER_HALFSIZE] = pixel;
}
if (threadIdx.x > xBlockLimit_max) {
int i = index(clamp(x + FILTER_HALFSIZE,width), clamp(y,height), width);
unsigned int pixel = input[i];
sharedPixels[threadIdx.x + 2*FILTER_HALFSIZE][threadIdx.y + FILTER_HALFSIZE] = pixel;
}
if (threadIdx.y < yBlockLimit_min) {
int i = index(clamp(x,width), clamp(y - FILTER_HALFSIZE,height), width);
unsigned int pixel = input[i];
sharedPixels[threadIdx.x + FILTER_HALFSIZE][threadIdx.y] = pixel;
}
if (threadIdx.y > yBlockLimit_max) {
int i = index(clamp(x,width), clamp(y + FILTER_HALFSIZE,height), width);
unsigned int pixel = input[i];
sharedPixels[threadIdx.x + FILTER_HALFSIZE][threadIdx.y + 2*FILTER_HALFSIZE] = pixel;
}
int i = index(x, y, width);
unsigned int pixel = input[i];
sharedPixels[threadIdx.x + FILTER_HALFSIZE][threadIdx.y + FILTER_HALFSIZE] = pixel;

__syncthreads();

if((x<width) && (y<height))
{
const int color_tid = y * width + x;
float windowMedian[MAX_WINDOW*MAX_WINDOW];
int windowElements = 0;

for (int x_iter = 0; x_iter < FILTER_SIZE; x_iter ++)
{
for (int y_iter = 0; y_iter < FILTER_SIZE; y_iter++)
{
if (0<=x_iter && x_iter < width && 0 <= y_iter && y_iter < height)
{
windowMedian[windowElements++] = sharedPixels[threadIdx.x + x_iter][threadIdx.y + y_iter];
}
}
}
sort_bubble(windowMedian,windowElements);
output[color_tid] = windowMedian[windowElements/2];
}
}