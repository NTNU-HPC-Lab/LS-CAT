#include "includes.h"
__global__ void kernel_add_wavelet ( float *g_u2, float wavelets, const int nx, const int ny, const int ngpus)
{
// global grid idx for (x,y) plane
int ipos = (ngpus == 2 ? ny - 10 : ny / 2 - 10);
unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
unsigned int idx = ipos * nx + ix;

if(ix == nx / 2) g_u2[idx] += wavelets;
}