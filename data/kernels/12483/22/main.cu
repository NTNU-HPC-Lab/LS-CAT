#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "field_summary.cu"
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
const int x_inner = 1;
const int y_inner = 1;
const int halo_depth = 1;
const double *volume = NULL;
cudaMalloc(&volume, XSIZE*YSIZE);
const double *density = NULL;
cudaMalloc(&density, XSIZE*YSIZE);
const double *energy0 = NULL;
cudaMalloc(&energy0, XSIZE*YSIZE);
const double *u = NULL;
cudaMalloc(&u, XSIZE*YSIZE);
double *vol_out = NULL;
cudaMalloc(&vol_out, XSIZE*YSIZE);
double *mass_out = NULL;
cudaMalloc(&mass_out, XSIZE*YSIZE);
double *ie_out = NULL;
cudaMalloc(&ie_out, XSIZE*YSIZE);
double *temp_out = NULL;
cudaMalloc(&temp_out, XSIZE*YSIZE);
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
field_summary<<<gridBlock,threadBlock>>>(x_inner,y_inner,halo_depth,volume,density,energy0,u,vol_out,mass_out,ie_out,temp_out);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
field_summary<<<gridBlock,threadBlock>>>(x_inner,y_inner,halo_depth,volume,density,energy0,u,vol_out,mass_out,ie_out,temp_out);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
field_summary<<<gridBlock,threadBlock>>>(x_inner,y_inner,halo_depth,volume,density,energy0,u,vol_out,mass_out,ie_out,temp_out);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}