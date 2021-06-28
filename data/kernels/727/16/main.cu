#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "conductance_calculate_postsynaptic_current_injection_kernel.cu"
#include<chrono>
#include<iostream>
using namespace std;
using namespace std::chrono;
int blocks_[20][2] = {{8,8},{16,16},{24,24},{32,32},{1,64},{1,128},{1,192},{1,256},{1,320},{1,384},{1,448},{1,512},{1,576},{1,640},{1,704},{1,768},{1,832},{1,896},{1,960},{1,1024}};
int matrices_[7][2] = {{240,240},{496,496},{784,784},{1016,1016},{1232,1232},{1680,1680},{2024,2024}};
int main(int argc, char **argv) {
cudaSetDevice(0);
char* p;int matrix_len=strtol(argv[1], &p, 10);
for(int matrix_looper=0;matrix_looper<matrix_len;matrix_looper++){
for(int block_looper=0;block_looper<20;block_looper++){
int XSIZE=matrices_[matrix_looper][0],YSIZE=matrices_[matrix_looper][1],BLOCKX=blocks_[block_looper][0],BLOCKY=blocks_[block_looper][1];
float *decay_term_values = NULL;
cudaMalloc(&decay_term_values, XSIZE*YSIZE);
float *reversal_values = NULL;
cudaMalloc(&reversal_values, XSIZE*YSIZE);
int num_decay_terms = 1;
int *synapse_decay_values = NULL;
cudaMalloc(&synapse_decay_values, XSIZE*YSIZE);
float *neuron_wise_conductance_traces = NULL;
cudaMalloc(&neuron_wise_conductance_traces, XSIZE*YSIZE);
float *d_neurons_current_injections = NULL;
cudaMalloc(&d_neurons_current_injections, XSIZE*YSIZE);
float *d_membrane_potentials_v = NULL;
cudaMalloc(&d_membrane_potentials_v, XSIZE*YSIZE);
float timestep = 1;
size_t total_number_of_neurons = 1;
int iXSIZE= XSIZE;
int iYSIZE= YSIZE;
while(iXSIZE%BLOCKX!=0)
{
iXSIZE++;
}
while(iYSIZE%BLOCKY!=0)
{
iYSIZE++;
}
dim3 gridBlock(iXSIZE/BLOCKX, iYSIZE/BLOCKY);
dim3 threadBlock(BLOCKX, BLOCKY);
cudaFree(0);
conductance_calculate_postsynaptic_current_injection_kernel<<<gridBlock,threadBlock>>>(decay_term_values,reversal_values,num_decay_terms,synapse_decay_values,neuron_wise_conductance_traces,d_neurons_current_injections,d_membrane_potentials_v,timestep,total_number_of_neurons);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
conductance_calculate_postsynaptic_current_injection_kernel<<<gridBlock,threadBlock>>>(decay_term_values,reversal_values,num_decay_terms,synapse_decay_values,neuron_wise_conductance_traces,d_neurons_current_injections,d_membrane_potentials_v,timestep,total_number_of_neurons);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
conductance_calculate_postsynaptic_current_injection_kernel<<<gridBlock,threadBlock>>>(decay_term_values,reversal_values,num_decay_terms,synapse_decay_values,neuron_wise_conductance_traces,d_neurons_current_injections,d_membrane_potentials_v,timestep,total_number_of_neurons);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}