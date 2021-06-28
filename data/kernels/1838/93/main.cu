#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "flow_absolute_residual_scalable_GPU.cu"
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
float *d_abs_res = NULL;
cudaMalloc(&d_abs_res, XSIZE*YSIZE);
const float2 *d_flow_compact = NULL;
cudaMalloc(&d_flow_compact, XSIZE*YSIZE);
const float *d_Zbuffer_flow_compact = NULL;
cudaMalloc(&d_Zbuffer_flow_compact, XSIZE*YSIZE);
const int *d_ind_flow_Zbuffer = NULL;
cudaMalloc(&d_ind_flow_Zbuffer, XSIZE*YSIZE);
const unsigned int *d_valid_flow_Zbuffer = NULL;
cudaMalloc(&d_valid_flow_Zbuffer, XSIZE*YSIZE);
float fx = 1;
float fy = 1;
float ox = 1;
float oy = 1;
int n_rows = 1;
int n_cols = 1;
int n_valid_flow_Zbuffer = 1;
const int *d_offset_ind = NULL;
cudaMalloc(&d_offset_ind, XSIZE*YSIZE);
const int *d_segment_translation_table = NULL;
cudaMalloc(&d_segment_translation_table, XSIZE*YSIZE);
float w_flow = 1;
float w_ar_flow = 1;
const float *d_dTR = NULL;
cudaMalloc(&d_dTR, XSIZE*YSIZE);
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
flow_absolute_residual_scalable_GPU<<<gridBlock,threadBlock>>>(d_abs_res,d_flow_compact,d_Zbuffer_flow_compact,d_ind_flow_Zbuffer,d_valid_flow_Zbuffer,fx,fy,ox,oy,n_rows,n_cols,n_valid_flow_Zbuffer,d_offset_ind,d_segment_translation_table,w_flow,w_ar_flow,d_dTR);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
flow_absolute_residual_scalable_GPU<<<gridBlock,threadBlock>>>(d_abs_res,d_flow_compact,d_Zbuffer_flow_compact,d_ind_flow_Zbuffer,d_valid_flow_Zbuffer,fx,fy,ox,oy,n_rows,n_cols,n_valid_flow_Zbuffer,d_offset_ind,d_segment_translation_table,w_flow,w_ar_flow,d_dTR);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
flow_absolute_residual_scalable_GPU<<<gridBlock,threadBlock>>>(d_abs_res,d_flow_compact,d_Zbuffer_flow_compact,d_ind_flow_Zbuffer,d_valid_flow_Zbuffer,fx,fy,ox,oy,n_rows,n_cols,n_valid_flow_Zbuffer,d_offset_ind,d_segment_translation_table,w_flow,w_ar_flow,d_dTR);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}