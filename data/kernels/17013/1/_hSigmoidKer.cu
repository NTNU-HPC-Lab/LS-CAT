#include "includes.h"
__global__ void _hSigmoidKer(float const *in, float *out, int size) {
int index = threadIdx.x + blockIdx.x * blockDim.x;
if (index >= size)
return ;

if (in[index] > 3 )
out[index] = 1;
else if (in[index] < -3)
out[index] = 0;
else
out[index] = (in[index] + 3)/6;
}