#include "includes.h"

namespace ann {








// CUDA2





}

__global__ void kernel_calc_gL( int layer_id, int *l, int *s, float *z_arr, float *a_arr, float *t_arr, float *gjl ){

volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

int neuron_count = l[layer_id];

if(idx >= neuron_count-1) return;

float z = z_arr[s[layer_id] + idx];
float tmp = 1 + expf(-z);
float f_deriv=expf(-z) / (tmp*tmp);

gjl[s[layer_id] + idx] = f_deriv*(a_arr[s[layer_id] + idx] - t_arr[idx]);
}