#include "includes.h"
__global__ void dsa_sumup_kernal(float* results, const int inx, const int iny)
{
int tidx = threadIdx.x;
int bd = blockDim.x;
int size = iny*(inx/2 + 1);

float dot = 0.0f; float vweight = 0.0f; float power = 0.0f; float power2 = 0.0f;
for (int i = 0; i < (inx/2 + 1); i++) {
int idx = i*bd + tidx;
dot += results[idx];
vweight += results[size + idx];
power += results[2*size + idx];
power2 += results[3*size + idx];
}

results[tidx] = dot;
results[size + tidx] = vweight;
results[2*size + tidx] = power;
results[3*size + tidx] = power2;

}