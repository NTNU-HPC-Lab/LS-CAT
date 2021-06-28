#include "includes.h"
__global__ void EFD_1dBM( int size, float *d_val_n, float *d_val_npo, float Pu, float Pm, float Pd ){
int i = threadIdx.x + blockDim.x * blockIdx.x;
if (i < size)
{
d_val_npo[i] = Pu * d_val_n[i + 1] + Pm * d_val_n[i] + Pd * d_val_n[i - 1];

if (i == 0)
{
d_val_npo[i] = d_val_npo[1];
}
else if (i == size - 1)
{
d_val_npo[i] = d_val_npo[i - 1];
}
}
}