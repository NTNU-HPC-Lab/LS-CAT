#include "includes.h"
__global__ void sobelFilterShared(unsigned char *data, unsigned char *result, int width, int height){
// Data cache: threadIdx.x , threadIdx.y
const int n = Mask_size / 2;
__shared__ int s_data[BLOCKSIZE + Mask_size * 2 ][BLOCKSIZE + Mask_size * 2];

// global mem address of the current thread in the whole grid
const int pos = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * width + blockIdx.y * blockDim.y * width;

// load cache (32x32 shared memory, 16x16 threads blocks)
// each threads loads four values from global memory into shared mem
// if in image area, get value in global mem, else 0
int x, y; // image based coordinate

// original image based coordinate
const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
const int y0 = threadIdx.y + blockIdx.y * blockDim.y;

// case1: upper left
x = x0 - n;
y = y0 - n;
if ( x < 0 || y < 0 )
s_data[threadIdx.y][threadIdx.x] = 0;
else
s_data[threadIdx.y][threadIdx.x] = *(data + pos - n - (width * n));

// case2: upper right
x = x0 + n;
y = y0 - n;
if ( x > (width - 1) || y < 0 )
s_data[threadIdx.y][threadIdx.x + blockDim.x] = 0;
else
s_data[threadIdx.y][threadIdx.x + blockDim.x] = *(data + pos + n - (width * n));

// case3: lower left
x = x0 - n;
y = y0 + n;
if (x < 0 || y > (height - 1))
s_data[threadIdx.y + blockDim.y][threadIdx.x] = 0;
else
s_data[threadIdx.y + blockDim.y][threadIdx.x] = *(data + pos - n + (width * n));

// case4: lower right
x = x0 + n;
y = y0 + n;
if ( x > (width - 1) || y > (height - 1))
s_data[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = 0;
else
s_data[threadIdx.y + blockDim.y][threadIdx.x + blockDim.x] = *(data + pos + n + (width * n));

__syncthreads();

// convolution
int sum = 0;
x = n + threadIdx.x;
y = n + threadIdx.y;
for (int i = - n; i <= n; i++)
for (int j = - n; j <= n; j++)
sum += s_data[y + i][x + j] * Global_Mask[n + i] * Global_Mask[n + j];

result[pos] = sum;
}