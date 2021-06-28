#include "includes.h"

__constant__ float *c_Kernel;

__global__ void compare(float *d_ip_v, float *d_ip_ir, int len) {

const int X = blockIdx.x * blockDim.x + threadIdx.x;

if (X < len)
{
d_ip_v[X] = (abs(d_ip_v[X]) > abs(d_ip_ir[X])) ? d_ip_v[X] : d_ip_ir[X];
}

}