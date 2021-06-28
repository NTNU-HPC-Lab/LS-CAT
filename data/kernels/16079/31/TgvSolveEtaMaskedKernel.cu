#include "includes.h"
__global__ void TgvSolveEtaMaskedKernel(float* mask, float alpha0, float alpha1, float* atensor, float *btensor, float* ctensor, float* etau, float* etav1, float* etav2, int width, int height, int stride)
{
int iy = blockIdx.y * blockDim.y + threadIdx.y;        // current row
int ix = blockIdx.x * blockDim.x + threadIdx.x;        // current column

if ((iy >= height) && (ix >= width)) return;
int pos = ix + iy * stride;
if (mask[pos] == 0.0f) return;

float a = atensor[pos];
float b = btensor[pos];
float c = ctensor[pos];

etau[pos] = (a*a + b * b + 2 * c*c + (a + c)*(a + c) + (b + c)*(b + c)) * (alpha1 * alpha1);
etav1[pos] = (alpha1 * alpha1)*(b * b + c * c) + 4 * alpha0 * alpha0;
etav2[pos] = (alpha1 * alpha1)*(a * a + c * c) + 4 * alpha0 * alpha0;
}