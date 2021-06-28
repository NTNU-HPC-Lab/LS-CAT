#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "sum_S_calc.cu"
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
float *S_calcc = NULL;
cudaMalloc(&S_calcc, XSIZE*YSIZE);
float *f_ptxc = NULL;
cudaMalloc(&f_ptxc, XSIZE*YSIZE);
float *f_ptyc = NULL;
cudaMalloc(&f_ptyc, XSIZE*YSIZE);
float *f_ptzc = NULL;
cudaMalloc(&f_ptzc, XSIZE*YSIZE);
float *S_calc = NULL;
cudaMalloc(&S_calc, XSIZE*YSIZE);
float *Aq = NULL;
cudaMalloc(&Aq, XSIZE*YSIZE);
float *q_S_ref_dS = NULL;
cudaMalloc(&q_S_ref_dS, XSIZE*YSIZE);
int num_q = 1;
int num_atom = 1;
int num_atom2 = 1;
float alpha = 2;
float k_chi = 1;
float *sigma2 = NULL;
cudaMalloc(&sigma2, XSIZE*YSIZE);
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
sum_S_calc<<<gridBlock,threadBlock>>>(S_calcc,f_ptxc,f_ptyc,f_ptzc,S_calc,Aq,q_S_ref_dS,num_q,num_atom,num_atom2,alpha,k_chi,sigma2);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
sum_S_calc<<<gridBlock,threadBlock>>>(S_calcc,f_ptxc,f_ptyc,f_ptzc,S_calc,Aq,q_S_ref_dS,num_q,num_atom,num_atom2,alpha,k_chi,sigma2);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
sum_S_calc<<<gridBlock,threadBlock>>>(S_calcc,f_ptxc,f_ptyc,f_ptzc,S_calc,Aq,q_S_ref_dS,num_q,num_atom,num_atom2,alpha,k_chi,sigma2);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}