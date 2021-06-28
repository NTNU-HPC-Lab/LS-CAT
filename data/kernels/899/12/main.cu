#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "gpu_copy_velocity.cu"
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
const int num_atoms = 1;
const int *g_group_contents = NULL;
cudaMalloc(&g_group_contents, XSIZE*YSIZE);
const double *g_vx_i = NULL;
cudaMalloc(&g_vx_i, XSIZE*YSIZE);
const double *g_vy_i = NULL;
cudaMalloc(&g_vy_i, XSIZE*YSIZE);
const double *g_vz_i = NULL;
cudaMalloc(&g_vz_i, XSIZE*YSIZE);
double *g_vx_o = NULL;
cudaMalloc(&g_vx_o, XSIZE*YSIZE);
double *g_vy_o = NULL;
cudaMalloc(&g_vy_o, XSIZE*YSIZE);
double *g_vz_o = NULL;
cudaMalloc(&g_vz_o, XSIZE*YSIZE);
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
gpu_copy_velocity<<<gridBlock,threadBlock>>>(num_atoms,g_group_contents,g_vx_i,g_vy_i,g_vz_i,g_vx_o,g_vy_o,g_vz_o);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
gpu_copy_velocity<<<gridBlock,threadBlock>>>(num_atoms,g_group_contents,g_vx_i,g_vy_i,g_vz_i,g_vx_o,g_vy_o,g_vz_o);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
gpu_copy_velocity<<<gridBlock,threadBlock>>>(num_atoms,g_group_contents,g_vx_i,g_vy_i,g_vz_i,g_vx_o,g_vy_o,g_vz_o);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}