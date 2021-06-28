#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "conductance_move_spikes_towards_synapses_kernel.cu"
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
int *d_spikes_travelling_to_synapse = NULL;
cudaMalloc(&d_spikes_travelling_to_synapse, XSIZE*YSIZE);
float current_time_in_seconds = 1;
int *circular_spikenum_buffer = NULL;
cudaMalloc(&circular_spikenum_buffer, XSIZE*YSIZE);
int *spikeid_buffer = NULL;
cudaMalloc(&spikeid_buffer, XSIZE*YSIZE);
int bufferloc = 1;
int buffersize = XSIZE*YSIZE;
int total_number_of_synapses = 1;
float *d_time_of_last_spike_to_reach_synapse = NULL;
cudaMalloc(&d_time_of_last_spike_to_reach_synapse, XSIZE*YSIZE);
int *postsynaptic_neuron_indices = NULL;
cudaMalloc(&postsynaptic_neuron_indices, XSIZE*YSIZE);
float *neuron_wise_conductance_trace = NULL;
cudaMalloc(&neuron_wise_conductance_trace, XSIZE*YSIZE);
int *synaptic_decay_id = NULL;
cudaMalloc(&synaptic_decay_id, XSIZE*YSIZE);
int total_number_of_neurons = 1;
float *d_synaptic_efficacies_or_weights = NULL;
cudaMalloc(&d_synaptic_efficacies_or_weights, XSIZE*YSIZE);
float *d_biological_conductance_scaling_constants_lambda = NULL;
cudaMalloc(&d_biological_conductance_scaling_constants_lambda, XSIZE*YSIZE);
float timestep = 1;
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
conductance_move_spikes_towards_synapses_kernel<<<gridBlock,threadBlock>>>(d_spikes_travelling_to_synapse,current_time_in_seconds,circular_spikenum_buffer,spikeid_buffer,bufferloc,buffersize,total_number_of_synapses,d_time_of_last_spike_to_reach_synapse,postsynaptic_neuron_indices,neuron_wise_conductance_trace,synaptic_decay_id,total_number_of_neurons,d_synaptic_efficacies_or_weights,d_biological_conductance_scaling_constants_lambda,timestep);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
conductance_move_spikes_towards_synapses_kernel<<<gridBlock,threadBlock>>>(d_spikes_travelling_to_synapse,current_time_in_seconds,circular_spikenum_buffer,spikeid_buffer,bufferloc,buffersize,total_number_of_synapses,d_time_of_last_spike_to_reach_synapse,postsynaptic_neuron_indices,neuron_wise_conductance_trace,synaptic_decay_id,total_number_of_neurons,d_synaptic_efficacies_or_weights,d_biological_conductance_scaling_constants_lambda,timestep);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
conductance_move_spikes_towards_synapses_kernel<<<gridBlock,threadBlock>>>(d_spikes_travelling_to_synapse,current_time_in_seconds,circular_spikenum_buffer,spikeid_buffer,bufferloc,buffersize,total_number_of_synapses,d_time_of_last_spike_to_reach_synapse,postsynaptic_neuron_indices,neuron_wise_conductance_trace,synaptic_decay_id,total_number_of_neurons,d_synaptic_efficacies_or_weights,d_biological_conductance_scaling_constants_lambda,timestep);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}