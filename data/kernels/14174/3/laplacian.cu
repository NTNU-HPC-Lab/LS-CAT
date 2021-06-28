#include "includes.h"
__global__ void laplacian(float *dst, const float *src, const size_t width, const size_t height, const size_t pixelsPerThread)
{
const size_t col  = (blockIdx.x * blockDim.x + threadIdx.x) % width;
const size_t crow = (blockIdx.x * blockDim.x + threadIdx.x) / width * pixelsPerThread;

if (col >= width || crow >= height)
return;

const size_t srow = crow + 1;
const size_t erow = min((unsigned int)(crow + pixelsPerThread - 1), (unsigned int)(height - 1));

// First element

const size_t firstIdx = crow * width + col;

dst[firstIdx] = src[firstIdx];

if (crow + 1 <  height) dst[firstIdx] -= 0.25f * src[firstIdx + width]; // S
if (crow     >= 1)      dst[firstIdx] -= 0.25f * src[firstIdx - width]; // N
if (col + 1  <  width)  dst[firstIdx] -= 0.25f * src[firstIdx + 1]; // E
if (col      >= 1)      dst[firstIdx] -= 0.25f * src[firstIdx - 1]; // W

// Inner elements

for (int row = srow; row < erow; ++row)
{
const size_t cIdx = row * width + col;

// C, S, N (always exist)
dst[cIdx] = src[cIdx] - 0.25f * (src[cIdx + width] + src[cIdx - width]);

if (col + 1 < width) dst[cIdx] -= 0.25f * src[cIdx + 1]; // E
if (col     >= 1)    dst[cIdx] -= 0.25f * src[cIdx - 1]; // W
}

if (erow <= crow)
return;

// Last element

const size_t lastIdx = erow * width + col;

dst[lastIdx] = src[lastIdx] - 0.25f * src[lastIdx - width]; // C, N

if (erow + 1 <  height) dst[lastIdx] -= 0.25f * src[lastIdx + width]; // S
if (col + 1  <  width)  dst[lastIdx] -= 0.25f * src[lastIdx + 1]; // E
if (col      >= 1)      dst[lastIdx] -= 0.25f * src[lastIdx - 1]; // W
}