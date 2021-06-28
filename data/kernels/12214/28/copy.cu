#include "includes.h"
__global__ void copy( float *v4, const float *v3, const int n ) {
for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) {
v4[i*8+0] = v3[i*6+0];
v4[i*8+1] = v3[i*6+1];
v4[i*8+2] = v3[i*6+2];
v4[i*8+4] = v3[i*6+3];
v4[i*8+5] = v3[i*6+4];
v4[i*8+6] = v3[i*6+5];
}
}