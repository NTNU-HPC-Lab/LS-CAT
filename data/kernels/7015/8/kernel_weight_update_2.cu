#include "includes.h"

namespace ann {








// CUDA2





}

__global__ void kernel_weight_update_2( int layer_id, int *l, int *s_ext, int *sw_ext, float *z_ext_arr, float *a_ext_arr, float *t_arr, float *gjl_ext, float *w_ext_arr, float *dw_ext_arr, float eta, float alpha ){

int idx = threadIdx.y + blockDim.y*blockIdx.y;
int h = blockDim.x;
int pidx=threadIdx.x;


int neuron_count = l[layer_id];
int neuron_count_next = l[layer_id+1];

if(idx >= neuron_count) return;

float a = a_ext_arr[s_ext[layer_id] + idx];

int index0 = s_ext[layer_id + 1] + pidx;
int index1 = sw_ext[layer_id] + idx*(neuron_count_next - 1) + pidx;
for(int k = pidx; k < neuron_count_next-1; k+=h){

float grad = a*gjl_ext[index0];
index0 += h;
float dw = dw_ext_arr[index1] = -eta*grad + alpha*dw_ext_arr[index1];

w_ext_arr[index1] += dw;


index1 += h;

}
}