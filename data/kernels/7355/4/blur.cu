#include "includes.h"
__global__ void blur( float * input, float * output, int  height, int width)
{

int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
if(x<height && y<width)
{
for(int k=0;k<3;k++)
{
float sum=0;
int count=0;
for(int i=x-BLUR_SIZE; i<= x+BLUR_SIZE; i++)
{
for(int j= y-BLUR_SIZE; j<=y+BLUR_SIZE;j++)
{
if(i>=0 && i<height && j>=0 && j<width)
{
count++;
sum+=input[3*(i*width+j)+k];
}
}
}
output[3*(x*width+y)+k]=sum/count;
}
}
else
return ;
}