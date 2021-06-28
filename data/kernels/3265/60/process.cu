#include "includes.h"
__global__ void process(int N_step, int N_inst, float *input, float *output){
int g_id = blockIdx.x * blockDim.x + threadIdx.x;
if(g_id >= N_inst) return;
float ans = 0.;
for(int t=0;t<N_step;++t){
for(int i=0;i<12;++i){
ans += input[(i+t)%VEC_SIZE + VEC_SIZE * g_id];
}
}
output[g_id] = ans;
return;
}