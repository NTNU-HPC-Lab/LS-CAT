#include "includes.h"
__global__ void UpdateExtNeuron(float *port_input_pt, float *port_value_pt, int n_node, int n_var, int n_port_var, int n_port)
{
int i_thread = threadIdx.x + blockIdx.x * blockDim.x;
if (i_thread<n_node*n_port) {
int i_port = i_thread%n_port;
int i_node = i_thread/n_port;
float *pip = port_input_pt + i_node*n_var + n_port_var*i_port;
//printf("port %d node %d pip %f\n", i_port, i_node, *pip);
port_value_pt[i_node*n_var + n_port_var*i_port]
= *pip;
*pip = 0.0;
}
}