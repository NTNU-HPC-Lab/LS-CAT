#include "includes.h"

namespace ann {








// CUDA2





}

__global__ void kernel_feedforward_2( int layer_id, int *l, int *s_ext, int *sw_ext, float *z_ext_arr, float *a_ext_arr, float *w_ext_arr ){

extern __shared__ int sm[];
float *sm_z = (float*)&sm[0];

int h = blockDim.x;
int h2 = blockDim.y;


int lidx = threadIdx.x;
int pidx = threadIdx.y;
int idx = threadIdx.x + blockDim.x*blockIdx.x;

int neuron_count = l[layer_id];
int neuron_count_prev = l[layer_id-1];

//printf("layer = %d idx = %d count = %d\n", layer_id, idx, neuron_count-1);
if(idx >= neuron_count-1) return;

float z = 0;
int index0=sw_ext[layer_id-1];
int index1=s_ext[layer_id-1];
for(int k = pidx; k < neuron_count_prev; k+=h2){
z += w_ext_arr[index0 + k*(neuron_count - 1) + idx]*a_ext_arr[index1 + k];
}

sm_z[pidx*h + lidx] = z;


__syncthreads();

if(pidx == 0){
z = 0;
for(int i = 0; i < h2; i++)
z += sm_z[i*h + lidx];

z_ext_arr[s_ext[layer_id] + idx] = z;
float a = 1.0 / (1.0 + expf(-z));
a_ext_arr[s_ext[layer_id] + idx] = a;
}


// printf("index = %d z = %.5f\n", s[layer_id] + idx, z);
// printf("a = %.20f\n", a);
}