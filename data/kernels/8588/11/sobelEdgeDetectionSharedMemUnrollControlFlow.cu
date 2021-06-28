#include "includes.h"
__global__ void sobelEdgeDetectionSharedMemUnrollControlFlow(int *input, int *output, int width, int height, int thresh) {

unsigned int blockSize = 32;
static __shared__ int shMem[34][34];

int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

int xind = threadIdx.x + 1;
int yind = threadIdx.y + 1;

shMem[xind][yind] = input[width * j + i];

if ( i > 0 && j > 0 && i < width - 1 && j < height - 1)
{
if(threadIdx.x == 0)
shMem[xind-1][yind] = input[width * j + i-1];

if(threadIdx.y == 0)
shMem[xind][yind-1] = input[width * (j-1) + i];

if(threadIdx.x == blockSize+1)
shMem[xind+1][yind] = input[width * j + i+1];

if(threadIdx.y == blockSize+1)
shMem[xind][yind+1] = input[width * (j+1) + i];

if(threadIdx.x == 0 && threadIdx.y == 0)
shMem[xind-1][yind-1] = input[width * (j-1) + i-1];

if(threadIdx.x == blockSize+1 && threadIdx.y == 0)
shMem[xind+1][yind-1] = input[width * (j-1) + i+1];

if(threadIdx.x == 0 && threadIdx.y == blockSize+1)
shMem[xind-1][yind+1] = input[width * (j+1) + i-1];

if(threadIdx.x == blockSize+1 && threadIdx.y == blockSize+1)
shMem[xind+1][yind+1] = input[width * (j+1) + i+1];
}
__syncthreads();


int sum1 = 0, sum2 = 0, magnitude;
int num = 3;

for(int xind = 1; xind < num; xind++)
{
for(int yind = 1; yind < num; yind++)
{
sum1 = shMem[xind+1][yind-1] -     shMem[xind-1][yind-1]
+ 2 * shMem[xind+1][yind  ] - 2 * shMem[xind-1][yind  ]
+     shMem[xind+1][yind+1] -     shMem[xind-1][yind+1];

sum2 = shMem[xind-1][yind-1] + 2 * shMem[xind][yind-1] + shMem[xind+1][yind-1]
- shMem[xind-1][yind+1] - 2 * shMem[xind][yind+1] - shMem[xind+1][yind+1];

magnitude = sum1 * sum1 + sum2 * sum2;

if(magnitude > thresh)
output[(j + yind - 1) * width + (i + xind - 1)] = 255;
else
output[(j + yind - 1) * width + (i + xind - 1)] = 0;

}
}

}