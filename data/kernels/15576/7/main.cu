#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "tonemap.cu"
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
float *d_x = NULL;
cudaMalloc(&d_x, XSIZE*YSIZE);
float *d_y = NULL;
cudaMalloc(&d_y, XSIZE*YSIZE);
float *d_log_Y = NULL;
cudaMalloc(&d_log_Y, XSIZE*YSIZE);
float *d_cdf_norm = NULL;
cudaMalloc(&d_cdf_norm, XSIZE*YSIZE);
float *d_r_new = NULL;
cudaMalloc(&d_r_new, XSIZE*YSIZE);
float *d_g_new = NULL;
cudaMalloc(&d_g_new, XSIZE*YSIZE);
float *d_b_new = NULL;
cudaMalloc(&d_b_new, XSIZE*YSIZE);
float min_log_Y = 1;
float max_log_Y = 1;
float log_Y_range = 1;
int num_bins = 1;
int num_pixels_y = 1;
int num_pixels_x = 1;
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
tonemap<<<gridBlock,threadBlock>>>(d_x,d_y,d_log_Y,d_cdf_norm,d_r_new,d_g_new,d_b_new,min_log_Y,max_log_Y,log_Y_range,num_bins,num_pixels_y,num_pixels_x);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
tonemap<<<gridBlock,threadBlock>>>(d_x,d_y,d_log_Y,d_cdf_norm,d_r_new,d_g_new,d_b_new,min_log_Y,max_log_Y,log_Y_range,num_bins,num_pixels_y,num_pixels_x);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
tonemap<<<gridBlock,threadBlock>>>(d_x,d_y,d_log_Y,d_cdf_norm,d_r_new,d_g_new,d_b_new,min_log_Y,max_log_Y,log_Y_range,num_bins,num_pixels_y,num_pixels_x);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}