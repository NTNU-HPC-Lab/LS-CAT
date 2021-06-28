#include "includes.h"

namespace ann {








// CUDA2





}

__global__ void kernel_weight_update( int layer_id, int *l, int *s, int *sw, float *z_arr, float *a_arr, float *t_arr, float *gjl, float *w_arr, float *dw_arr, float eta, float alpha ){

volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

int neuron_count = l[layer_id];
int neuron_count_next = l[layer_id+1];

if(idx >= neuron_count) return;

float a = a_arr[s[layer_id] + idx];
for(int k = 0; k < neuron_count_next-1; k++){

float grad=/*a_arr[s[layer_id] + idx]*/a*gjl[s[layer_id + 1] + k];

dw_arr[sw[layer_id] + idx*(neuron_count_next - 1) + k]=
-eta*grad+
alpha*dw_arr[sw[layer_id] + idx*(neuron_count_next - 1) + k];

w_arr[sw[layer_id] + idx*(neuron_count_next - 1) + k]+=
dw_arr[sw[layer_id] + idx*(neuron_count_next - 1) + k];
}
}