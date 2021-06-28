#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "graph_determ_weights.cu"
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
unsigned int *contact_mat_cum_row_indexes = NULL;
cudaMalloc(&contact_mat_cum_row_indexes, XSIZE*YSIZE);
unsigned int *contact_mat_column_indexes = NULL;
cudaMalloc(&contact_mat_column_indexes, XSIZE*YSIZE);
float *contact_mat_values = NULL;
cudaMalloc(&contact_mat_values, XSIZE*YSIZE);
unsigned int rows = 1;
unsigned int values = 1;
float *immunities = NULL;
cudaMalloc(&immunities, XSIZE*YSIZE);
float *shedding_curve = NULL;
cudaMalloc(&shedding_curve, XSIZE*YSIZE);
unsigned int infection_length = 1;
float transmission_rate = 1;
int *infection_mat_values = NULL;
cudaMalloc(&infection_mat_values, XSIZE*YSIZE);
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
graph_determ_weights<<<gridBlock,threadBlock>>>(contact_mat_cum_row_indexes,contact_mat_column_indexes,contact_mat_values,rows,values,immunities,shedding_curve,infection_length,transmission_rate,infection_mat_values);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
graph_determ_weights<<<gridBlock,threadBlock>>>(contact_mat_cum_row_indexes,contact_mat_column_indexes,contact_mat_values,rows,values,immunities,shedding_curve,infection_length,transmission_rate,infection_mat_values);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
graph_determ_weights<<<gridBlock,threadBlock>>>(contact_mat_cum_row_indexes,contact_mat_column_indexes,contact_mat_values,rows,values,immunities,shedding_curve,infection_length,transmission_rate,infection_mat_values);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}