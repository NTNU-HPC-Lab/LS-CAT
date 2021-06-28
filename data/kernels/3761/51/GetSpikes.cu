#include "includes.h"
__global__ void GetSpikes(double *spike_array, int array_size, int n_port, int n_var, float *port_weight_arr, int port_weight_arr_step, int port_weight_port_step, float *port_input_arr, int port_input_arr_step, int port_input_port_step)
{
int i_array = threadIdx.x + blockIdx.x * blockDim.x;
if (i_array < array_size*n_port) {
int i_target = i_array % array_size;
int port = i_array / array_size;
int port_input = i_target*port_input_arr_step
+ port_input_port_step*port;
int port_weight = i_target*port_weight_arr_step
+ port_weight_port_step*port;
double d_val = (double)port_input_arr[port_input]
+ spike_array[i_array]
* port_weight_arr[port_weight];

port_input_arr[port_input] = (float)d_val;
}
}