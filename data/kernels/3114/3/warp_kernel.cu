#include "includes.h"
__global__ void warp_kernel(float* out, const float* in, const int* index,const float* weights,const int npixels,const int nchannels){
int pixel   = blockIdx.x * blockDim.x + threadIdx.x;
int channel = blockIdx.y * blockDim.y + threadIdx.y;
if( channel >= nchannels||pixel >= npixels)
return;
out[nchannels*pixel+channel]=in[nchannels*index[4*pixel]+channel]*weights[4*pixel]
+in[nchannels*index[4*pixel+1]+channel]*weights[4*pixel+1]
+in[nchannels*index[4*pixel+2]+channel]*weights[4*pixel+2]
+in[nchannels*index[4*pixel+3]+channel]*weights[4*pixel+3];
}