#include "includes.h"






















__device__ curandState randomStates[256];



__global__ void Softmax_Rowwise_M_Backward(const float* origin, const float* adjoint, const float* primal, const float* prevMaxs, const float* prevMaxIndices, const float* prevSums, float* out, const int rows, const int cols, const int cols2, const int n)
{
extern __shared__ float sdata[];
float* rowBuffer = sdata;
float* originData = &sdata[blockDim.x];
float* adjointData = &sdata[blockDim.x * 2];
float* primalData = &sdata[blockDim.x * 3];
float* outData = &sdata[blockDim.x * 4];

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

float prevMax = prevMaxs[ri];
int prevMaxIndex = prevMaxIndices[ri];
float prevSum = prevSums[ri];

if (inData)
{
originData[ti] = origin[i];
adjointData[ti] = adjoint[i];
primalData[ti] = primal[i];
}

// Div_DM_D				DM (direct)						 D (indirect via Sum_DM)
rowBuffer[ti] = adjointData[ti] / prevSum + adjointData[ti] * (originData[ti] / (prevSum * prevSum));

// Exp_DM				DM (direct)
rowBuffer[ti] = rowBuffer[ti] * __expf(originData[ti] - prevMax);
outData[ti] = rowBuffer[ti];

__syncthreads();

// calculate each rows derivatives (in rowBuffer) sum
for (int offset = cols2 / 2; offset > 0; offset >>= 1)
{
if (tiLocal < offset)
{
float other = (ti + offset) / cols == riLocal ? rowBuffer[ti + offset] : 0.0f;

rowBuffer[ti] = rowBuffer[ti] + other;
}

__syncthreads();
}

// Item_DM		D (indirect via Max op via Sub_DM_D op (left part for DM is just passthrough of gradient, so nothing to do there))
if (tiLocal == prevMaxIndex)
{
outData[ti] = outData[ti] - rowBuffer[riLocal * cols];
}

if (inData)
{
out[i] = outData[ti];
}
}