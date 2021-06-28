#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "binning_mix.cu"
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
float *xd_real = NULL;
cudaMalloc(&xd_real, XSIZE*YSIZE);
float *yd_real = NULL;
cudaMalloc(&yd_real, XSIZE*YSIZE);
float *zd_real = NULL;
cudaMalloc(&zd_real, XSIZE*YSIZE);
float *xd_sim = NULL;
cudaMalloc(&xd_sim, XSIZE*YSIZE);
float *yd_sim = NULL;
cudaMalloc(&yd_sim, XSIZE*YSIZE);
float *zd_sim = NULL;
cudaMalloc(&zd_sim, XSIZE*YSIZE);
float *ZY = NULL;
cudaMalloc(&ZY, XSIZE*YSIZE);
int lines_number_1 = 1;
int lines_number_2 = 1;
int points_per_degree = 1;
int number_of_degrees = 1;
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
binning_mix<<<gridBlock,threadBlock>>>(xd_real,yd_real,zd_real,xd_sim,yd_sim,zd_sim,ZY,lines_number_1,lines_number_2,points_per_degree,number_of_degrees);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
binning_mix<<<gridBlock,threadBlock>>>(xd_real,yd_real,zd_real,xd_sim,yd_sim,zd_sim,ZY,lines_number_1,lines_number_2,points_per_degree,number_of_degrees);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
binning_mix<<<gridBlock,threadBlock>>>(xd_real,yd_real,zd_real,xd_sim,yd_sim,zd_sim,ZY,lines_number_1,lines_number_2,points_per_degree,number_of_degrees);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}