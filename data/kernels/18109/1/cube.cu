#include "includes.h"
__global__ void cube(double* d_out, double* d_in)
{
int idx = threadIdx.x;
double f = d_in[idx];
d_out[idx] = f*f*f;

}