#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "vec_computePSF_phaseN.cu"
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
int n = XSIZE*YSIZE;
double *kx = NULL;
cudaMalloc(&kx, XSIZE*YSIZE);
double *ky = NULL;
cudaMalloc(&ky, XSIZE*YSIZE);
double *kz = NULL;
cudaMalloc(&kz, XSIZE*YSIZE);
double *pupil = NULL;
cudaMalloc(&pupil, XSIZE*YSIZE);
double *phase = NULL;
cudaMalloc(&phase, XSIZE*YSIZE);
double dx = 1;
double dy = 1;
double dz = 1;
int *sparseIndexEvenDisk = NULL;
cudaMalloc(&sparseIndexEvenDisk, XSIZE*YSIZE);
int *sparseIndexOddDisk = NULL;
cudaMalloc(&sparseIndexOddDisk, XSIZE*YSIZE);
double *fft = NULL;
cudaMalloc(&fft, XSIZE*YSIZE);
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
vec_computePSF_phaseN<<<gridBlock,threadBlock>>>(n,kx,ky,kz,pupil,phase,dx,dy,dz,sparseIndexEvenDisk,sparseIndexOddDisk,fft);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
vec_computePSF_phaseN<<<gridBlock,threadBlock>>>(n,kx,ky,kz,pupil,phase,dx,dy,dz,sparseIndexEvenDisk,sparseIndexOddDisk,fft);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
vec_computePSF_phaseN<<<gridBlock,threadBlock>>>(n,kx,ky,kz,pupil,phase,dx,dy,dz,sparseIndexEvenDisk,sparseIndexOddDisk,fft);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}