#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "network_corr.cu"
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
float *templates = NULL;
cudaMalloc(&templates, XSIZE*YSIZE);
float *sum_square_template = NULL;
cudaMalloc(&sum_square_template, XSIZE*YSIZE);
int *moveout = NULL;
cudaMalloc(&moveout, XSIZE*YSIZE);
float *data = NULL;
cudaMalloc(&data, XSIZE*YSIZE);
float *weights = NULL;
cudaMalloc(&weights, XSIZE*YSIZE);
size_t step = 1;
size_t n_samples_template = 1;
size_t n_samples_data = 1;
size_t n_stations = 1;
size_t n_components = 1;
int chunk_offset = 1;
int chunk_size = XSIZE*YSIZE;
float *cc_mat = NULL;
cudaMalloc(&cc_mat, XSIZE*YSIZE);
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
network_corr<<<gridBlock,threadBlock>>>(templates,sum_square_template,moveout,data,weights,step,n_samples_template,n_samples_data,n_stations,n_components,chunk_offset,chunk_size,cc_mat);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
network_corr<<<gridBlock,threadBlock>>>(templates,sum_square_template,moveout,data,weights,step,n_samples_template,n_samples_data,n_stations,n_components,chunk_offset,chunk_size,cc_mat);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
network_corr<<<gridBlock,threadBlock>>>(templates,sum_square_template,moveout,data,weights,step,n_samples_template,n_samples_data,n_stations,n_components,chunk_offset,chunk_size,cc_mat);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}