#include "includes.h"
__global__ void init_and_update (float *values_d, int tpoints, int nsteps){
int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;

if(idx <= 1 || idx >= tpoints)
return;

float old_v, v, new_v;

float x, tmp;
tmp = tpoints - 1;
x = (float)(idx - 1) / tmp;

v = sin(2.0f * PI * x);
old_v = v;

for (int i = 1; i <= nsteps; i++){
new_v = (2.0f * v) - old_v + (0.09f * (-2.0f * v));
old_v = v;
v = new_v;
}

values_d[idx] = v;

}