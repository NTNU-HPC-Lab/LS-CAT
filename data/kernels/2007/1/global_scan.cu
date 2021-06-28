#include "includes.h"



cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);





__global__ void global_scan(float* d_out, float* d_in)
{
int index = threadIdx.x;
float out = 0.00f;
d_out[index] = d_in[index];
__syncthreads();

for (int i = 1; i < sizeof(d_in); i*=2)
{
if (index - i >= 0)
{
out = d_out[index] + d_out[index - i];
}
__syncthreads();

if (index - i >=0)
{
d_out[index] = out;
out = 0.0f;
}
}

}