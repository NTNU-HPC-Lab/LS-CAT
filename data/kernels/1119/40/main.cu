#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "cuSetupSincKernel_kernel.cu"
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
float *r_filter_ = NULL;
cudaMalloc(&r_filter_, XSIZE*YSIZE);
const int i_filtercoef_ = 1;
const float r_soff_ = 1;
const float r_wgthgt_ = 1;
const int i_weight_ = 1;
const float r_soff_inverse_ = 1;
const float r_beta_ = 1;
const float r_decfactor_inverse_ = 1;
const float r_relfiltlen_inverse_ = 1;
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
cuSetupSincKernel_kernel<<<gridBlock,threadBlock>>>(r_filter_,i_filtercoef_,r_soff_,r_wgthgt_,i_weight_,r_soff_inverse_,r_beta_,r_decfactor_inverse_,r_relfiltlen_inverse_);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
cuSetupSincKernel_kernel<<<gridBlock,threadBlock>>>(r_filter_,i_filtercoef_,r_soff_,r_wgthgt_,i_weight_,r_soff_inverse_,r_beta_,r_decfactor_inverse_,r_relfiltlen_inverse_);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
cuSetupSincKernel_kernel<<<gridBlock,threadBlock>>>(r_filter_,i_filtercoef_,r_soff_,r_wgthgt_,i_weight_,r_soff_inverse_,r_beta_,r_decfactor_inverse_,r_relfiltlen_inverse_);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}