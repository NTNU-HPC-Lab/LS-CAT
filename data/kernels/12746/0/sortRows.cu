#include "includes.h"

#define THRESHOLD 10010000

__device__ int cudaGetNextInColor(int *image, int x, int row, int imageWidth, int color){
for (int i = x + 1; i < imageWidth; ++i)
{
if(THRESHOLD >= (color - image[row*imageWidth + i])){
return i-1;
}
}
return imageWidth - 1;
}
__device__ int cudaGetFirstNotInColor(int *image, int x, int row, int imageWidth, int color){
for (int i = x; i < imageWidth; ++i)
{
if(THRESHOLD < (color - image[row*imageWidth + i])){
return i;
}
}
return -1;
}
__device__ void bubbleSort(int *pixelsToSort, int length){
for(int i = 0; i < length; i++ )
{
for(int j = 0; j < length-1; j++)
{
if( pixelsToSort[j] > pixelsToSort[j+1]){
int tmp = pixelsToSort[j];
pixelsToSort[j] = pixelsToSort[j+1];
pixelsToSort[j+1] = tmp;
}
}
}
}
__global__ void sortRows(int *image, int imageHeight, int imageWidth, int colorMode){
int row = blockIdx.x * blockDim.x + threadIdx.x;
if(row < imageHeight)
{
int startingX = 0;
int finishX = 0;

while(finishX < imageWidth)
{
startingX = cudaGetFirstNotInColor(image, startingX, row, imageWidth, colorMode);
finishX = cudaGetNextInColor(image, startingX, row, imageWidth, colorMode);

if(startingX < 0)
break;

bubbleSort(&image[row*imageWidth + startingX], finishX - startingX);

startingX = finishX + 1;
}
}
}