#include "includes.h"

namespace ann {








// CUDA2





}

__global__ void kernel_calc_gjL_2( int layer_id, int *l, int *s_ext, int *sw_ext, float *z_ext_arr, float *a_ext_arr, float *t_arr, float *gjl_ext, float *w_ext_arr ){

int idx = threadIdx.y + blockDim.y*blockIdx.y;
int h = blockDim.x;
int pidx = threadIdx.y;
int lidx = threadIdx.x;

extern __shared__ int sm[];
float *sm_g = (float*)&sm[0];


int neuron_count = l[layer_id];
int neuron_count_next = l[layer_id+1];

if(idx >= neuron_count-1) return;

float sum = 0;
for (int k = lidx; k < neuron_count_next-1; k+=h) {
sum += w_ext_arr[sw_ext[layer_id] + idx*(l[layer_id + 1] - 1) + k] * gjl_ext[s_ext[layer_id + 1] + k];
}

sm_g[pidx*h + lidx] = sum;

__syncthreads();

if(lidx == 0){
float z = z_ext_arr[s_ext[layer_id] + idx];
float tmp = 1 + expf(-z);
float f_deriv = expf(-z) / (tmp*tmp);

sum = 0;
for(int i = 0; i < h; i++)
sum += sm_g[pidx*h + i];


gjl_ext[s_ext[layer_id] + idx] = f_deriv*sum;
}
}