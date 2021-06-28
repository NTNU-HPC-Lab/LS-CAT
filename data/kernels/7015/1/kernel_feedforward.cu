#include "includes.h"

namespace ann {








// CUDA2





}

__global__ void kernel_feedforward( int layer_id, int *l, int *s, int *sw, float *z_arr, float *a_arr, float *w_arr ){
volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;

int neuron_count = l[layer_id];
int neuron_count_prev = l[layer_id-1];

//printf("layer = %d idx = %d count = %d\n", layer_id, idx, neuron_count-1);
if(idx >= neuron_count-1) return;

float z = 0;
for(int k = 0; k < neuron_count_prev; k++){
z += w_arr[sw[layer_id-1] + k*(neuron_count - 1) + idx]*a_arr[s[layer_id-1] + k];
// printf("w_arr[%d] * a_arr[%d] = %.20f\n",
// 		sw[layer_id-1] + k*(neuron_count - 1) + idx ,
// 		s[layer_id-1] + k,
// 		w_arr[sw[layer_id-1] + k*(neuron_count - 1) + idx]*a_arr[s[layer_id-1] + k]);
// printf("%.10f * %.10f = %.10f\n", w_arr[sw[layer_id-1] + k*(neuron_count - 1) + idx ],
// 		a_arr[s[layer_id-1] + k],
// 		w_arr[sw[layer_id-1] + k*(neuron_count - 1) + idx]*a_arr[s[layer_id-1] + k]
// 	);

}

z_arr[s[layer_id] + idx] = z;
float a = 1.0 / (1.0 + expf(-z));
a_arr[s[layer_id] + idx] = a;
// printf("index = %d z = %.5f\n", s[layer_id] + idx, z);
// printf("a = %.20f\n", a);
}