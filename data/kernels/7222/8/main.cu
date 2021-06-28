#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "back.cu"
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
double *h_out_d = NULL;
cudaMalloc(&h_out_d, XSIZE*YSIZE);
double *weights_out_d = NULL;
cudaMalloc(&weights_out_d, XSIZE*YSIZE);
double *weights_h_d = NULL;
cudaMalloc(&weights_h_d, XSIZE*YSIZE);
double *weights_in_d = NULL;
cudaMalloc(&weights_in_d, XSIZE*YSIZE);
double *outputs_d = NULL;
cudaMalloc(&outputs_d, XSIZE*YSIZE);
double *deltas_h_d = NULL;
cudaMalloc(&deltas_h_d, XSIZE*YSIZE);
double *deltas_h_new_d = NULL;
cudaMalloc(&deltas_h_new_d, XSIZE*YSIZE);
double *deltas_o_d = NULL;
cudaMalloc(&deltas_o_d, XSIZE*YSIZE);
double *weights_in_delta_d = NULL;
cudaMalloc(&weights_in_delta_d, XSIZE*YSIZE);
double *weights_out_delta_d = NULL;
cudaMalloc(&weights_out_delta_d, XSIZE*YSIZE);
double *weights_h_delta_d = NULL;
cudaMalloc(&weights_h_delta_d, XSIZE*YSIZE);
int height = YSIZE;
int inputs = 1;
int outputs = 1;
int layers = 1;
double *training_in_d = NULL;
cudaMalloc(&training_in_d, XSIZE*YSIZE);
double *training_out_d = NULL;
cudaMalloc(&training_out_d, XSIZE*YSIZE);
int sample = 1;
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
back<<<gridBlock,threadBlock>>>(h_out_d,weights_out_d,weights_h_d,weights_in_d,outputs_d,deltas_h_d,deltas_h_new_d,deltas_o_d,weights_in_delta_d,weights_out_delta_d,weights_h_delta_d,height,inputs,outputs,layers,training_in_d,training_out_d,sample);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
back<<<gridBlock,threadBlock>>>(h_out_d,weights_out_d,weights_h_d,weights_in_d,outputs_d,deltas_h_d,deltas_h_new_d,deltas_o_d,weights_in_delta_d,weights_out_delta_d,weights_h_delta_d,height,inputs,outputs,layers,training_in_d,training_out_d,sample);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
back<<<gridBlock,threadBlock>>>(h_out_d,weights_out_d,weights_h_d,weights_in_d,outputs_d,deltas_h_d,deltas_h_new_d,deltas_o_d,weights_in_delta_d,weights_out_delta_d,weights_h_delta_d,height,inputs,outputs,layers,training_in_d,training_out_d,sample);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}