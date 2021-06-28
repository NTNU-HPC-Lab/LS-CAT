#include "includes.h"

//macro to check return value of the cuda runtime call and exits
//if call failed
__global__ void anyMethod(unsigned char* buff , unsigned char* buffer_out , int w , int h)
{
int x = blockIdx.x * blockDim.x +threadIdx.x ;
int y = blockIdx.y * blockDim.y +threadIdx.y;
int width = w , height = h;

if((x>=0 && x < width) && (y>=0 && y<height))
{
int hx = -buff[width*(y-1) + (x-1)] + buff[width*(y-1)+(x+1)]
-2*buff[width*(y)+(x-1)] + 2* buff[width*(y)+(x+1)]
-buff[width*(y+1)+(x-1)] + buff[width*(y+1)+(x+1)];

int vx = buff[width*(y-1)+(x-1)] +2*buff[width*(y-1)+(x+1)] +buff[width*(y-1)+(x+1)]
-buff[width*(y+1)+(x-1)] -2* buff[width*(y+1)+(x)] - buff[width*(y+1)+(x+1)];
//this is the main part changed to get the sort of tie dye effect for at least
//the first part of the picture
hx = hx*4;
vx = vx/5;

int val = (int)sqrt((float)(hx) * (float)(hx) + (float)(vx) * (float)(vx));

buffer_out[y * width + x] = (unsigned char) val;
}
}