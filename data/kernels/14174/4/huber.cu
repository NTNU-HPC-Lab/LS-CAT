#include "includes.h"
__global__ void huber(float *a, const size_t width, const size_t height, const float alpha, const float strength, const size_t pixelsPerThread, float *f)
{
const size_t col  = (blockIdx.x * blockDim.x + threadIdx.x) % width;
const size_t crow = (blockIdx.x * blockDim.x + threadIdx.x) / width * pixelsPerThread;

if (col >= width || crow >= height)
return;

const size_t erow = min((unsigned int)(crow + pixelsPerThread), (unsigned int)height);

const float alpha2 = alpha * alpha;

float colF = 0.0f;

for (size_t row = crow; row < erow; ++row)
{
const size_t idx = row * width + col;

// Pseudo-Huber loss function
const float root = sqrtf(1.0f + a[idx]*a[idx] / alpha2);
colF += alpha2 * (root - 1.0f);
a[idx] *= strength / root;
}

colF *= strength;
f[blockIdx.x * blockDim.x + threadIdx.x] = colF;
}