#include "includes.h"

#define TILE_WIDTH 32
#define TILE_HEIGHT 32
#define FSize 256
//void convolution(int *InputImage,int width,int height,int *filter,int filterWidth,,int padding,int *result);
using namespace std;

__global__ void MatrixMultiple(int *InputImage,int width,int height,int *filter,int filterWidth,int *featureMap)
{
/* get global row col */
int Row=blockIdx.y*TILE_HEIGHT+threadIdx.y;
int Col=blockIdx.x*TILE_WIDTH+threadIdx.x;
int value=0;
int feathreMapwidth=width-filterWidth+1;
if(Row*width+Col<width*height)
{
for(int i=0;i<filterWidth;i++)
{
for(int j=0;j<filterWidth;j++)
{
value+=filter[i*filterWidth+j]* InputImage[(Row+i)*width+Col+j];
}
}
//printf("%d %d\n",Row*width+Col,value);

featureMap[feathreMapwidth*Row+Col]=value;
}
//printf("%d %d\n",Row*width+Col,value);
}