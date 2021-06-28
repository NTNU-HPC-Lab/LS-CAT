#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "rgb_to_xyY.cu"
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
float *d_r = NULL;
cudaMalloc(&d_r, XSIZE*YSIZE);
float *d_g = NULL;
cudaMalloc(&d_g, XSIZE*YSIZE);
float *d_b = NULL;
cudaMalloc(&d_b, XSIZE*YSIZE);
float *d_x = NULL;
cudaMalloc(&d_x, XSIZE*YSIZE);
float *d_y = NULL;
cudaMalloc(&d_y, XSIZE*YSIZE);
float *d_log_Y = NULL;
cudaMalloc(&d_log_Y, XSIZE*YSIZE);
float delta = 1;
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
rgb_to_xyY<<<gridBlock,threadBlock>>>(d_r,d_g,d_b,d_x,d_y,d_log_Y,delta,num_pixels_y,num_pixels_x);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
rgb_to_xyY<<<gridBlock,threadBlock>>>(d_r,d_g,d_b,d_x,d_y,d_log_Y,delta,num_pixels_y,num_pixels_x);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
rgb_to_xyY<<<gridBlock,threadBlock>>>(d_r,d_g,d_b,d_x,d_y,d_log_Y,delta,num_pixels_y,num_pixels_x);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}