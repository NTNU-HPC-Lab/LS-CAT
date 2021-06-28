#include "includes.h"
__global__ void calc_psf_hat(float* d_psf, float *d_psf_hat, int psf_rows, int psf_cols)
{
int row = threadIdx.y + blockIdx.y * blockDim.y;
int col = threadIdx.x + blockIdx.x * blockDim.x;
if(row < psf_rows && col < psf_cols)
{
int index = (psf_rows - row - 1) * psf_cols + psf_cols - col - 1;
d_psf_hat[index] = d_psf[row * psf_cols + col];
//        if(d_psf_hat[index] > 0)
//            printf("psf_hat[%d] = %f - psf[%d] = %f\n", index,d_psf_hat[index], row * psf_cols + col, d_psf[row * psf_cols + col]);
}
}