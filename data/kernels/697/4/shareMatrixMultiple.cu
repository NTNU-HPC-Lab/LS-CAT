#include "includes.h"

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define FSize 256
//void convolution(int *InputImage,int width,int height,int *filter,int filterWidth,,int padding,int *result);
using namespace std;

__global__ void shareMatrixMultiple(int *InputImage,int width,int height,int *filter,int filterWidth,int *featureMap)
{
extern __shared__ int tileImage[];

int Row=blockIdx.y*TILE_HEIGHT+threadIdx.y;
int Col=blockIdx.x*TILE_WIDTH+threadIdx.x;
int value=0;
int feathreMapwidth=width-filterWidth+1;
int shareWidth=(TILE_WIDTH+filterWidth-1);

tileImage[threadIdx.y*shareWidth+threadIdx.x]=InputImage[Row*width+Col];
if(threadIdx.x<filterWidth-1)
{
tileImage[threadIdx.y*shareWidth+threadIdx.x+TILE_WIDTH]=InputImage[Row*width+Col+TILE_WIDTH];
}
if(threadIdx.y<filterWidth-1)
{
tileImage[(threadIdx.y+TILE_HEIGHT)*shareWidth+threadIdx.x]=InputImage[(Row+TILE_HEIGHT)*width+Col];
}
if(threadIdx.x<filterWidth-1 && threadIdx.y<filterWidth-1)
{
tileImage[(threadIdx.y+TILE_HEIGHT)*shareWidth+threadIdx.x+TILE_WIDTH]=InputImage[(Row+TILE_HEIGHT)*width+Col+TILE_WIDTH];
}

__syncthreads();

if(Row*width+Col<width*height)
{
for(int i=0;i<filterWidth;i++)
{
for(int j=0;j<filterWidth;j++)
{
//value+=filter[i*filterWidth+j]* InputImage[(Row+i)*width+Col+j];
value+=filter[i*filterWidth+j]* tileImage[(threadIdx.y+i)*shareWidth+threadIdx.x+j];
}
}
//printf("%d %d\n",Row*width+Col,value);
featureMap[feathreMapwidth*Row+Col]=value;
}
}