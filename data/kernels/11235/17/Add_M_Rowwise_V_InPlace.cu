#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void Add_M_Rowwise_V_InPlace(const float* a, const int rows, const int cols, const int cols2, float* b, const int n)
{
extern __shared__ float sdata[];

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
sdata[ti] = x;

__syncthreads();

// calculate each rows derivatives (in sdata) sum
for (int offset = cols2 / 2; offset > 0; offset >>= 1)
{
if (tiLocal < offset)
{
float other = (ti + offset) / cols == riLocal ? sdata[ti + offset] : 0.0f;

sdata[ti] = sdata[ti] + other;
}

__syncthreads();
}

if (tiLocal == 0)
{
b[ri] = b[ri] + sdata[riLocal * cols];
}
}