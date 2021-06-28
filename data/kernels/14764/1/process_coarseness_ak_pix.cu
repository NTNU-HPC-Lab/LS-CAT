#include "includes.h"
__device__ double efficientLocalMean_dev (const long x,const long y,const long k, double * input_img, int rowsize, int colsize) {
long k2 = k/2;

long dimx = rowsize;
long dimy = colsize;

//wanting average over area: (y-k2,x-k2) ... (y+k2-1, x+k2-1)
long starty = y-k2;
long startx = x-k2;
long stopy = y+k2-1;
long stopx = x+k2-1;

if (starty < 0) starty = 0;
if (startx < 0) startx = 0;
if (stopx > dimx-1) stopx = dimx-1;
if (stopy > dimy-1) stopy = dimy-1;

double unten, links, oben, obenlinks;

if (startx-1 < 0) links = 0;
else links = *(input_img+(stopy * dimx + startx-1));

if (starty-1 < 0) oben = 0;
else oben = *(input_img+((stopy-1) * dimx + startx));

if ((starty-1 < 0) || (startx-1 <0)) obenlinks = 0;
else obenlinks = *(input_img+((stopy-1) * dimx + startx-1));

unten = *(input_img+(stopy * dimx + startx));

long counter = (stopy-starty+1)*(stopx-startx+1);
return (unten-links-oben+obenlinks)/counter;
}
__global__ void process_coarseness_ak_pix(double * output_ak,double * input_img,int colsize, int rowsize,long lenOf_ak)
{
int index;
int y  = threadIdx.x + blockIdx.x * blockDim.x;
int x = threadIdx.y + blockIdx.y * blockDim.y;
if(y < (colsize) && x < (rowsize))
{
index = y * rowsize + x ;
output_ak[index] = efficientLocalMean_dev(x,y,lenOf_ak,input_img,rowsize,colsize);
}
}