#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "__pairmult2.cu"
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
int nrows = 1;
int bncols = 1;
int brows1 = 1;
int brows2 = 1;
float *A = NULL;
cudaMalloc(&A, XSIZE*YSIZE);
int lda = 1;
float *A2 = NULL;
cudaMalloc(&A2, XSIZE*YSIZE);
int lda2 = 1;
float *Bdata = NULL;
cudaMalloc(&Bdata, XSIZE*YSIZE);
int *Bir = NULL;
cudaMalloc(&Bir, XSIZE*YSIZE);
int *Bjc = NULL;
cudaMalloc(&Bjc, XSIZE*YSIZE);
int broff = 1;
int bcoff = 1;
float *C = NULL;
cudaMalloc(&C, XSIZE*YSIZE);
int ldc = 1;
int transpose = 1;
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
__pairmult2<<<gridBlock,threadBlock>>>(nrows,bncols,brows1,brows2,A,lda,A2,lda2,Bdata,Bir,Bjc,broff,bcoff,C,ldc,transpose);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
__pairmult2<<<gridBlock,threadBlock>>>(nrows,bncols,brows1,brows2,A,lda,A2,lda2,Bdata,Bir,Bjc,broff,bcoff,C,ldc,transpose);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
__pairmult2<<<gridBlock,threadBlock>>>(nrows,bncols,brows1,brows2,A,lda,A2,lda2,Bdata,Bir,Bjc,broff,bcoff,C,ldc,transpose);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}