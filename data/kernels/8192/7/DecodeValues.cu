#include "includes.h"
__global__  void DecodeValues(float* superposition, int symbolSize, float* output, float* reliability, int numOfValues, int squaredMode, float* dirX, float* dirY, float* negDirX, float* negDirY, float* originX, float* originY)
{
int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
+ blockDim.x*blockIdx.x				//blocks preceeding current block
+ threadIdx.x;

if (threadId >= numOfValues)
return;


output[threadId] = 0;
reliability[threadId] = 0;

float* dir = threadId == 0 ? dirX : dirY;
float* negDir = threadId == 0 ? negDirX : negDirY;
float* origin = threadId == 0 ? originX : originY;

for (int i = 0; i < symbolSize; i++)
{
// output  = s.d - s.n = s.dir
// one of the values s.d or s.n will be (very close to) zero
output[threadId] += superposition[i] * dir[i] - superposition[i] * negDir[i];
// rel	   = s.o
reliability[threadId] += superposition[i] * origin[i];
}

// rel     = s.o + s.dir
reliability[threadId] += fabs(output[threadId]);
// output  = s.dir / (s.o + s.dir)
output[threadId] /= reliability[threadId];

// Since s = dir*t + o*(1-t) + noise, we get
// s.dir   = dir.dir*t + o.dir*(1-t) + dir.noise = t + 0 + dir.noise
// s.o     = o.dir*t   + o.o*(1-t)   + o.noise   = 0 + (1-t) + o.noise
// output  = t + dir.noise / (1 + dir.noise + o.noise)
// Note that dir.noise and o.noise should be very close to zero.
// This should make the decoding more precise when noise has similar dot product to dir and o.
}