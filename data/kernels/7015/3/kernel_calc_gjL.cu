#include "includes.h"

namespace ann {








// CUDA2





}

__global__ void kernel_calc_gjL( int layer_id, int *l, int *s, int *sw, float *z_arr, float *a_arr, float *t_arr, float *gjl, float *w_arr ){

volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

int neuron_count = l[layer_id];
int neuron_count_next = l[layer_id+1];

if(idx >= neuron_count-1) return;

//float f_deriv=expf(-z_arr[s[layer_id] + idx]) / powf((1 + expf(-z_arr[s[layer_id] + idx])),2.0f);
float z = z_arr[s[layer_id] + idx];
float tmp = 1 + expf(-z);
float f_deriv=expf(-z) / (tmp*tmp);


float sum = 0;
for (int k = 0; k < neuron_count_next-1; k++) {
sum += w_arr[sw[layer_id] + idx*(l[layer_id + 1] - 1) + k] * gjl[s[layer_id + 1] + k];
}

gjl[s[layer_id] + idx] = f_deriv*sum;
// printf("Kernelis %d - %.20f\n", s[layer_id] + idx, gjl[s[layer_id] + idx]);
}