#include "includes.h"
__global__ void process(int N_step, int N_inst, float *input, float *output){
int b_id = blockIdx.x, t_id = threadIdx.x;
if(b_id >= N_inst) return;
__shared__ float ans;
float val;
if(!t_id) ans = 0;
if(t_id < VEC_SIZE) val = input[VEC_SIZE * b_id + t_id];
__syncthreads();
for(int t=0;t<N_step;++t){
int start = t%VEC_SIZE;
if(t_id >= start && t_id < min(start + 12, VEC_SIZE)) atomicAdd(&ans, val);
if(start + 12 > VEC_SIZE && t_id < start + 12 - VEC_SIZE) atomicAdd(&ans, val);
__syncthreads();
}
output[b_id] = ans;
return;
}