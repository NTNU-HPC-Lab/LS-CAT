#include "includes.h"
__global__ void convolution_kernel(unsigned char *input_img, unsigned char *output_img, int height, int width)
{

__shared__ unsigned char input_shared[W][W]; //Shared Memory required for a tile and its halo elements(3 channels)

int chan;
for(chan=0;chan<3;chan++)    //3 Channel Image
{
int tx = threadIdx.x;
int ty = threadIdx.y;

int output_row = blockIdx.x*TILE_WIDTH + tx;
int output_col = blockIdx.y*TILE_WIDTH + ty;

int input_row = output_row - MASK_WIDTH/2;
int input_col = output_col - MASK_WIDTH/2;

if((input_row >= 0) && (input_row < height) && (input_col >= 0) && (input_col < width))
{
input_shared[tx][ty] = input_img[(input_row*width + input_col)*3 + chan];
}
else
{
if(input_row<0 && input_col<0)
{
input_shared[tx][ty] = input_img[chan];
}
else if(input_row<0 && input_col<width)
{
input_shared[tx][ty] = input_img[3*input_col + chan];
}
else if(input_row<0)
{
input_shared[tx][ty] = input_img[3*(width-1) + chan];
}
else if(input_row<height && input_col<0)
{
input_shared[tx][ty] = input_img[input_row*width*3 + chan];
}
else if(input_row<height && input_col>width)
{
input_shared[tx][ty] = input_img[(input_row*width +width-1)*3 + chan];
}
else if(input_row>height && input_col<0)
{
input_shared[tx][ty] = input_img[width*(height-1)*3 + chan];
}
else if(input_row>height && input_col<width)
{
input_shared[tx][ty] = input_img[(width*(height-1)+input_col)*3 + chan];
}
else
{
input_shared[tx][ty] = input_img[(width*(height-1) + (width-1))*3 + chan];
}
}

__syncthreads();

int i;
if(tx<TILE_WIDTH && ty<TILE_WIDTH)
{
int j;
int freq[256];

for(i=0;i<256;i++)freq[i]=0;

for(i=0;i<MASK_WIDTH;i++)
{
for(j=0;j<MASK_WIDTH;j++)
{
freq[input_shared[tx+i][ty+j]]++;
}
}
j=0;
for(i=0;i<256;i++)
{
j=j+freq[i];
if(j>((MASK_WIDTH*MASK_WIDTH)/2))break;
}
}

if(output_row<height && output_col<width)
{
output_img[(output_row*width + output_col)*3 + chan] = i;
}
__syncthreads();
}

}