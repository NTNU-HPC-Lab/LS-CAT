#include "includes.h"
__global__ void sobelFilterShared2(unsigned char *data, unsigned char *result, int width, int height){
// Data cache: threadIdx.x , threadIdx.y
int ty = threadIdx.y;
int tx = threadIdx.x;

// shared memory represented here by 1D array
// each thread loads two values from global memory into shared mem
const int n = Mask_size / 2;
__shared__ int s_data[BLOCKSIZE * (BLOCKSIZE + Mask_size * 2)];

// global mem address of the current thread in the whole grid
const int pos = tx + blockIdx.x * blockDim.x + ty * width + blockIdx.y * blockDim.y * width;

// load cache (32x32 shared memory, 16x16 threads blocks)
// each threads loads four values from global memory into shared mem
// if in image area, get value in global mem, else 0
int y; // image based coordinate

// original image based coordinate
const int y0 = ty + blockIdx.y * blockDim.y;
const int shift = ty * (BLOCKSIZE);

// case1: upper left
y = y0 - n;
if ( y < 0 )
s_data[tx + shift] = 0;
else
s_data[tx + shift] = data[ pos - (width * n)];

// case2: lower
y = y0 - n;
const int shift1 = shift + blockDim.y * BLOCKSIZE;

if ( y > height - 1)
s_data[tx + shift1] = 0;
else
s_data[tx + shift1] = data[pos +  (width * n)];

__syncthreads();

// convolution
int sum = 0;
for (int i = 0; i <= n*2; i++)
sum += s_data[tx + (ty+i) * BLOCKSIZE] * Global_Mask[i];

result[pos] = sum;
}