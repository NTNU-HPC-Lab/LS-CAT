#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "DeformablePSROIPoolForwardKernel.cu"
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
const int count = 1;
const double *bottom_data = NULL;
cudaMalloc(&bottom_data, XSIZE*YSIZE);
const double spatial_scale = 1;
const int channels = 1;
const int height = 1;
const int width = 1;
const int pooled_height = 1;
const int pooled_width = 1;
const double *bottom_rois = NULL;
cudaMalloc(&bottom_rois, XSIZE*YSIZE);
const double *bottom_trans = NULL;
cudaMalloc(&bottom_trans, XSIZE*YSIZE);
const int no_trans = 1;
const double trans_std = 1;
const int sample_per_part = 1;
const int output_dim = 1;
const int group_size = 1;
const int part_size = 1;
const int num_classes = 1;
const int channels_each_class = 1;
double *top_data = NULL;
cudaMalloc(&top_data, XSIZE*YSIZE);
double *top_count = NULL;
cudaMalloc(&top_count, XSIZE*YSIZE);
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
DeformablePSROIPoolForwardKernel<<<gridBlock,threadBlock>>>(count,bottom_data,spatial_scale,channels,height,width,pooled_height,pooled_width,bottom_rois,bottom_trans,no_trans,trans_std,sample_per_part,output_dim,group_size,part_size,num_classes,channels_each_class,top_data,top_count);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
DeformablePSROIPoolForwardKernel<<<gridBlock,threadBlock>>>(count,bottom_data,spatial_scale,channels,height,width,pooled_height,pooled_width,bottom_rois,bottom_trans,no_trans,trans_std,sample_per_part,output_dim,group_size,part_size,num_classes,channels_each_class,top_data,top_count);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
DeformablePSROIPoolForwardKernel<<<gridBlock,threadBlock>>>(count,bottom_data,spatial_scale,channels,height,width,pooled_height,pooled_width,bottom_rois,bottom_trans,no_trans,trans_std,sample_per_part,output_dim,group_size,part_size,num_classes,channels_each_class,top_data,top_count);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}