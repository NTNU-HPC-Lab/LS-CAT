#include "includes.h"
__global__ void d_putgaps(float *sne7, float *snaw, int *aw2ali, const int snno)
{
//sino index
int sni = threadIdx.x + blockIdx.y*blockDim.x;

//sino bin index
int awi = blockIdx.x;

if (sni<snno) {
sne7[aw2ali[awi] * snno + sni] = snaw[awi*snno + sni];
}
}