#include "includes.h"

namespace ann {








// CUDA2





}

__global__ void kernel_calc_gL_2( int layer_id, int *l, int *s_ext, float *z_ext_arr, float *a_ext_arr, float *t_arr, float *gjl_ext ){

volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

int neuron_count = l[layer_id];

if(idx >= neuron_count-1) return;

float z = z_ext_arr[s_ext[layer_id] + idx];
float tmp = 1 + expf(-z);
float f_deriv=expf(-z) / (tmp*tmp);

gjl_ext[s_ext[layer_id] + idx] = f_deriv*(a_ext_arr[s_ext[layer_id] + idx] - t_arr[idx]);
}