#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void Softmax_Rowwise_M(const float* a, float* maxPerRow, float* maxPerRowIndices, float* sumPerRow, const int rows, const int cols, const int cols2, float* out, const int n)
{
extern __shared__ float sdata[];
float* rowBuffer = &sdata[blockDim.x];

int rowsPerBlock = blockDim.x / cols;
int usedPerBlock = rowsPerBlock * cols;
int unusedPerBlock = blockDim.x - usedPerBlock;

int ti = threadIdx.x;
int blockId = blockIdx.y * gridDim.x + blockIdx.x;
int i = blockId * blockDim.x + threadIdx.x - (unusedPerBlock * blockId);
int ri = i / cols;
int riLocal = ri % rowsPerBlock;
int tiLocal = ti - riLocal * cols;
bool inData = i < n && ti < usedPerBlock;

float x = 0.0f;
if (inData)
{
x = a[i];
}
sdata[ti] = rowBuffer[ti] = x;

__syncthreads();

// find each rows max value
for (int offset = cols2 / 2; offset > 0; offset >>= 1)
{
if (tiLocal < offset)
{
float currentMax = rowBuffer[ti];
float other = (ti + offset) / cols == riLocal ? rowBuffer[ti + offset] : 0.0f;

rowBuffer[ti] = other > currentMax ? other : currentMax;
}

__syncthreads();
}

// subtract each value from that row's maximum
if (inData)
{
sdata[ti] = __expf(sdata[ti] - rowBuffer[riLocal * cols]);

if (tiLocal == 0)
{
maxPerRow[ri] = rowBuffer[riLocal * cols];
}
}
rowBuffer[ti] = sdata[ti];

__syncthreads();

// write out max index
if (maxPerRow[ri] == a[i])
{
maxPerRowIndices[ri] = tiLocal;
}

// calculate each rows sum
for (int offset = cols2 / 2; offset > 0; offset >>= 1)
{
if (tiLocal < offset)
{
float other = (ti + offset) / cols == riLocal ? rowBuffer[ti + offset] : 0.0f;

rowBuffer[ti] = rowBuffer[ti] + other;
}

__syncthreads();
}

if (inData)
{
out[i] = sdata[ti] / rowBuffer[riLocal * cols];

if (tiLocal == 0)
{
sumPerRow[ri] = rowBuffer[riLocal * cols];
}
}
}