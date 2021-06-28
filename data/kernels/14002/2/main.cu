#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "opt_cond_itr.cu"
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
int num_train_cases = 1;
double *opt_cond = NULL;
cudaMalloc(&opt_cond, XSIZE*YSIZE);
double alpha_high = 2;
double alpha_high_prev = 2;
int high_label = 1;
int high_indx = 1;
double alpha_low = 2;
double alpha_low_prev = 2;
int low_label = 1;
int low_indx = 1;
double *kernel_val_mat = NULL;
cudaMalloc(&kernel_val_mat, XSIZE*YSIZE);
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
opt_cond_itr<<<gridBlock,threadBlock>>>(num_train_cases,opt_cond,alpha_high,alpha_high_prev,high_label,high_indx,alpha_low,alpha_low_prev,low_label,low_indx,kernel_val_mat);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
opt_cond_itr<<<gridBlock,threadBlock>>>(num_train_cases,opt_cond,alpha_high,alpha_high_prev,high_label,high_indx,alpha_low,alpha_low_prev,low_label,low_indx,kernel_val_mat);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
opt_cond_itr<<<gridBlock,threadBlock>>>(num_train_cases,opt_cond,alpha_high,alpha_high_prev,high_label,high_indx,alpha_low,alpha_low_prev,low_label,low_indx,kernel_val_mat);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}