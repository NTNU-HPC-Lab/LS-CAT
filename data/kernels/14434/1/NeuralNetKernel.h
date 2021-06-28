#ifndef NERUALNETKERNEL_H
#define NERUALNETKERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <algorithm>

extern float * g_weights_d;

void start_cuda(int size, float * weights);

void copy_weights(int size, float * weights);

void print_weights(int size);

void end_cuda();

										//number neurons per sweeper, number neurons per layer, number neurons per input layer, number neurons per output layer, number sweepers total, number of weights per sweeper, number of HIDDEN layers, bias of neural net, and response of sigmoid function
void call_cuda_neural_net(int num_per_sweeper, int num_per_layer, int num_per_input, int num_per_output, int num_sweepers, int num_weights, int num_layers, float response, float *inputs, float * outputs);

__global__ void cuda_neural_net(float *Weights_D, int num_per_sweeper, int num_per_layer, int num_per_input, int num_per_output, int num_weights, int num_layers, float response, float *inputs_d, float *outputs_d);

#endif //NERUALNETKERNEL_H