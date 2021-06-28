#include "includes.h"
__global__ void pythagoras(unsigned char* Gx, unsigned char* Gy, unsigned char* G, unsigned char* theta)
{
int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

float af = float(Gx[idx]);
float bf = float(Gy[idx]);

G[idx] = (unsigned char)sqrtf(af * af + bf * bf);
theta[idx] = (unsigned char)atan2f(af, bf)*63.994;

}