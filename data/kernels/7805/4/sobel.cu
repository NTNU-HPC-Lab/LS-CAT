#include "includes.h"
__global__ void sobel(unsigned char *output, unsigned char *input, int width, int height)
{
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;

if (y >= height || x >= width)
return;

const int BLOCK_SIZE = 16;

// Where does our data start
int blockStartIndexX = blockIdx.x * blockDim.x - 1;
int blockStartIndexY = blockIdx.y * blockDim.y - 1;

// Clamp to edge
if (blockStartIndexX < 0)
blockStartIndexX = 0;

if (blockStartIndexX >= width)
blockStartIndexX = blockDim.x - 1;

if (blockStartIndexY < 0)
blockStartIndexY = 0;

if (blockStartIndexY >= height)
blockStartIndexY = blockDim.y - 1;

// Shared Data
__shared__ unsigned char pixels[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

// Where is our data
unsigned char* cacheInput = input + (blockStartIndexX + blockStartIndexY * width);

// Linear index (16x16 -> 0..255)
int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;

int maxLoadSizeBytes = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2); // 18x18 Block -> 324 Bytes
int maxIndexBytes = maxLoadSizeBytes / sizeof(short); // 18x18 Block -> Index 162

if (threadIndex < maxIndexBytes)
{
// Calculate offset
int offsetInBytes = threadIndex * sizeof(short);

int block_half = (BLOCK_SIZE + 2) / 2;
int byteRow = offsetInBytes / (BLOCK_SIZE + 2);
int byteCol = threadIndex % block_half * 2;

int offset = byteCol + byteRow * width;
//int offsetBuffer = byteCol + byteRow * (BLOCK_SIZE + 2);

// Copy Data
unsigned char* toLoad = cacheInput + offset;
/**(&pixels[0][0] + offsetBuffer) = *toLoad;
*(&pixels[0][0] + offsetBuffer + 1) = *(toLoad + 1);*/

pixels[byteRow][byteCol] = *toLoad;
pixels[byteRow][byteCol + 1] = *(toLoad + 1);
}

__syncthreads();


// Sobel weights
float weightsX[9] = { -1, -2, -1,
0,  0,  0,
1,  2,  1 };

float weightsY[9] = { -1,  0,  1,
-2,  0,  2,
-1,  0,  1 };

int offsetY[9] = { -1,  -1,  -1,
0,   0,   0,
1,   1,   1 };

int offsetX[9] = { -1,   0,   1,
-1,   0,   1,
-1,   0,   1 };


float pointX = 0.f;
float pointY = 0.f;

#pragma unroll
for (int i = 0; i < 9; i++)
{
int indexX = threadIdx.x + 1 + offsetX[i];
int indexY = threadIdx.y + 1 + offsetY[i];

unsigned char pixel = pixels[indexY][indexX];
pointX += pixel * weightsX[i];
pointY += pixel * weightsY[i];
}


// Do Sobel here!
int index = x + y * width;
unsigned char * outputData = output + index;
outputData[0] = sqrtf(pointX * pointX + pointY * pointY);
}