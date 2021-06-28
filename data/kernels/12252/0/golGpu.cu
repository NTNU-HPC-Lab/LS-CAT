#include "includes.h"

__global__ void golGpu(int height, int width, unsigned char* pBuffer1, unsigned char* pBuffer2){
int x = blockIdx.x * 2 + threadIdx.x;
int y = blockIdx.y * 2 + threadIdx.y;

int indx = x * height + y;

pBuffer2[indx] = pBuffer1[indx];

int num = 0;

if (x-1 >= 0 && x-1 < height && y >= 0 && y < width)
num += pBuffer1[(x-1) * height + y];

if (x+1 >= 0 && x+1 < height && y >= 0 && y < width)
num += pBuffer1[(x+1) * height + y];

if (x >= 0 && x < height && y-1 >= 0 && y-1 < width)
num += pBuffer1[x * height + (y-1)];

if (x >= 0 && x < height && y+1 >= 0 && y+1 < width)
num += pBuffer1[x * height + (y+1)];

if (x-1 >= 0 && x-1 < height && y-1 >= 0 && y-1 < width)
num += pBuffer1[(x-1) * height + (y-1)];

if (x-1 >= 0 && x-1 < height && y+1 >= 0 && y+1 < width)
num += pBuffer1[(x-1) * height + (y+1)];

if (x+1 >= 0 && x+1 < height && y-1 >= 0 && y-1 < width)
num += pBuffer1[(x+1) * height + (y-1)];

if (x+1 >= 0 && x+1 < height && y+1 >= 0 && y+1 < width)
num += pBuffer1[(x+1) * height + (y+1)];

if(num < 2)
pBuffer2[indx] = 0x0;

if(num > 3)
pBuffer2[indx] = 0x0;

if(num == 3 && !pBuffer1[indx])
pBuffer2[indx] = 0x1;
//return num;

}