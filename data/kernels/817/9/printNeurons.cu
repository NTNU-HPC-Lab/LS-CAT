#include "includes.h"
__global__ void printNeurons(int *d_neuron_outputs, float *d_weighted_sums){
int neuron = blockIdx.x*blockDim.x + threadIdx.x;
printf("neuron: %d, weighted sum: %.2f, output: %d\n", neuron, d_weighted_sums[neuron], d_neuron_outputs[neuron]);
}