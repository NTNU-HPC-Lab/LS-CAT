#include "includes.h"
__global__ void dwt_average(float *d_ip_v, float *d_ip_ir, int app_len) {

const int X = blockIdx.x * blockDim.x + threadIdx.x;

if (X < app_len)
{
d_ip_v[X] = (d_ip_v[X] + d_ip_ir[X]) / 2;
}

}