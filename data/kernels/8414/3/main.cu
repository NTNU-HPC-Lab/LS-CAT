#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "calculate_sumterm_part.cu"
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
double2 *Up = NULL;
cudaMalloc(&Up, XSIZE*YSIZE);
double2 *Vpl = NULL;
cudaMalloc(&Vpl, XSIZE*YSIZE);
const double2 *A_t = NULL;
cudaMalloc(&A_t, XSIZE*YSIZE);
const double *SR = NULL;
cudaMalloc(&SR, XSIZE*YSIZE);
const unsigned char *nonzero_midx1234s = NULL;
cudaMalloc(&nonzero_midx1234s, XSIZE*YSIZE);
const unsigned int N = 1;
const unsigned int M = 1;
const double SK_factor = 1;
const unsigned int NUM_NONZERO = 1;
const unsigned int NUM_MODES = 1;
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
calculate_sumterm_part<<<gridBlock,threadBlock>>>(Up,Vpl,A_t,SR,nonzero_midx1234s,N,M,SK_factor,NUM_NONZERO,NUM_MODES);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
calculate_sumterm_part<<<gridBlock,threadBlock>>>(Up,Vpl,A_t,SR,nonzero_midx1234s,N,M,SK_factor,NUM_NONZERO,NUM_MODES);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
calculate_sumterm_part<<<gridBlock,threadBlock>>>(Up,Vpl,A_t,SR,nonzero_midx1234s,N,M,SK_factor,NUM_NONZERO,NUM_MODES);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}