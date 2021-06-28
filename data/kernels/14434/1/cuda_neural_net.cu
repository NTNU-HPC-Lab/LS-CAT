#include "includes.h"
__global__ void cuda_neural_net(float *Weights_D, int num_per_sweeper, int num_per_layer, int num_per_input, int num_per_output, int num_weights, int num_layers, float response, float *inputs_d, float *outputs_d)
{

extern __shared__ float buffer[];

int start_of_weights = blockIdx.x * num_weights;
int start_of_hidden_layers = start_of_weights + (num_per_input * num_per_layer);


//input layer
buffer[threadIdx.x] = 0;
for (int i = 0; i < num_per_input; ++i)
{
buffer[threadIdx.x] += inputs_d[(blockIdx.x * num_per_input) + i] * Weights_D[start_of_weights + (threadIdx.x * num_per_input) + i];
}
buffer[threadIdx.x] = 1.0 / (1.0 + exp(-buffer[threadIdx.x] / response));
__syncthreads();

//subsequent hidden layers
float temp;

for (int i = 0; i < num_layers; ++i)
{
temp = 0;
for (int j = 0; j < num_per_layer; ++j)
{
temp += buffer[j] * Weights_D[start_of_hidden_layers + (num_per_layer * num_per_layer * i) + (num_per_layer * threadIdx.x) + j];
}
temp = 1.0 / (1.0 + exp(-temp / response));

__syncthreads();
buffer[threadIdx.x] = temp;
__syncthreads();
}

//output layer
if (threadIdx.x < num_per_output)
{
temp = 0;
for (int i = 0; i < num_per_layer; ++i)
{
temp += buffer[i] * Weights_D[start_of_hidden_layers + (num_per_layer * num_per_layer * num_layers) + (num_per_layer * threadIdx.x) + i];
}
temp = 1.0 / (1.0 + exp(-temp / response));

__syncthreads();

//copy the result back out to the outputs vector
outputs_d[(blockIdx.x * num_per_output) + threadIdx.x] = temp;
}

}