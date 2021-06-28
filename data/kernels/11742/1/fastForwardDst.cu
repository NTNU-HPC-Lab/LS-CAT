#include "includes.h"
__global__ void fastForwardDst(const int16_t* block, int16_t* coeff, int shift)  // input block, output coeff
{
int c[4];
int rnd_factor = 1 << (shift - 1);
int i = threadIdx.x;
// Intermediate Variables
c[0] = block[4 * i + 0] + block[4 * i + 3];
c[1] = block[4 * i + 1] + block[4 * i + 3];
c[2] = block[4 * i + 0] - block[4 * i + 1];
c[3] = 74 * block[4 * i + 2];

coeff[i] = (int16_t)((29 * c[0] + 55 * c[1] + c[3] + rnd_factor) >> shift);
coeff[4 + i] = (int16_t)((74 * (block[4 * i + 0] + block[4 * i + 1] - block[4 * i + 3]) + rnd_factor) >> shift);
coeff[8 + i] = (int16_t)((29 * c[2] + 55 * c[0] - c[3] + rnd_factor) >> shift);
coeff[12 + i] = (int16_t)((55 * c[2] - 29 * c[1] + c[3] + rnd_factor) >> shift);
}