#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "normal_eqs_flow_multicam_GPU.cu"
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
float *d_CO = NULL;
cudaMalloc(&d_CO, XSIZE*YSIZE);
float2 *d_flow_compact = NULL;
cudaMalloc(&d_flow_compact, XSIZE*YSIZE);
float *d_Zbuffer_flow_compact = NULL;
cudaMalloc(&d_Zbuffer_flow_compact, XSIZE*YSIZE);
int *d_ind_flow_Zbuffer = NULL;
cudaMalloc(&d_ind_flow_Zbuffer, XSIZE*YSIZE);
const float *d_focal_length = NULL;
cudaMalloc(&d_focal_length, XSIZE*YSIZE);
const float *d_nodal_point_x = NULL;
cudaMalloc(&d_nodal_point_x, XSIZE*YSIZE);
const float *d_nodal_point_y = NULL;
cudaMalloc(&d_nodal_point_y, XSIZE*YSIZE);
const int *d_n_rows = NULL;
cudaMalloc(&d_n_rows, XSIZE*YSIZE);
const int *d_n_cols = NULL;
cudaMalloc(&d_n_cols, XSIZE*YSIZE);
const int *d_n_values_flow = NULL;
cudaMalloc(&d_n_values_flow, XSIZE*YSIZE);
const int *d_start_ind_flow = NULL;
cudaMalloc(&d_start_ind_flow, XSIZE*YSIZE);
const int *d_pixel_ind_offset = NULL;
cudaMalloc(&d_pixel_ind_offset, XSIZE*YSIZE);
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
normal_eqs_flow_multicam_GPU<<<gridBlock,threadBlock>>>(d_CO,d_flow_compact,d_Zbuffer_flow_compact,d_ind_flow_Zbuffer,d_focal_length,d_nodal_point_x,d_nodal_point_y,d_n_rows,d_n_cols,d_n_values_flow,d_start_ind_flow,d_pixel_ind_offset);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
normal_eqs_flow_multicam_GPU<<<gridBlock,threadBlock>>>(d_CO,d_flow_compact,d_Zbuffer_flow_compact,d_ind_flow_Zbuffer,d_focal_length,d_nodal_point_x,d_nodal_point_y,d_n_rows,d_n_cols,d_n_values_flow,d_start_ind_flow,d_pixel_ind_offset);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
normal_eqs_flow_multicam_GPU<<<gridBlock,threadBlock>>>(d_CO,d_flow_compact,d_Zbuffer_flow_compact,d_ind_flow_Zbuffer,d_focal_length,d_nodal_point_x,d_nodal_point_y,d_n_rows,d_n_cols,d_n_values_flow,d_start_ind_flow,d_pixel_ind_offset);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}