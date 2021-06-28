#include "includes.h"
__global__ void change_theta(const int ncoord, const float3 *theta, float4 *thetax, float4 *thetay, float4 *thetaz) {

unsigned int pos = blockIdx.x*blockDim.x + threadIdx.x;
if (pos < ncoord) {
thetax[pos].x = theta[pos*4].x;
thetax[pos].y = theta[pos*4+1].x;
thetax[pos].z = theta[pos*4+2].x;
thetax[pos].w = theta[pos*4+3].x;

thetay[pos].x = theta[pos*4].y;
thetay[pos].y = theta[pos*4+1].y;
thetay[pos].z = theta[pos*4+2].y;
thetay[pos].w = theta[pos*4+3].y;

thetaz[pos].x = theta[pos*4].z;
thetaz[pos].y = theta[pos*4+1].z;
thetaz[pos].z = theta[pos*4+2].z;
thetaz[pos].w = theta[pos*4+3].z;
}

}