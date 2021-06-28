#include "includes.h"
__global__ void Solve_redblack1_Kernel(float* output, const float* input, int width, int height, int nChannels, int c, const float* weightdata, const float* weightx, const float* weighty, float lambda, float omega, bool redflag)
{
int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;

int x = bx*blockDim.x + tx;
int y = by*blockDim.y + ty;
if (x >= width || y >= height)
return;

if ((y + x) % 2 == redflag)
return;

int offset = y*width + x;
int slice = width*nChannels;
int offset_c = offset*nChannels + c;
float coeff = 0, sigma = 0, weight = 0;
if (y > 0)
{
weight = lambda*weighty[offset - width];
coeff += weight;
sigma += weight * output[offset_c - slice];
}
if (y < height - 1)
{
weight = lambda*weighty[offset];
coeff += weight;
sigma += weight*output[offset_c + slice];
}
if (x > 0)
{
weight = lambda*weightx[offset - 1];
coeff += weight;
sigma += weight*output[offset_c - nChannels];
}
if (x < width - 1)
{
weight = lambda*weightx[offset];
coeff += weight;
sigma += weight*output[offset_c + nChannels];
}
coeff += weightdata[offset];
sigma += weightdata[offset] * input[offset_c];
if (coeff > 0)
output[offset_c] = sigma / coeff*omega + output[offset_c] * (1 - omega);
}