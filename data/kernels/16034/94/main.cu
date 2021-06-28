#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "vec_computePSF_phaseNManywithOil_f.cu"
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
int sizePart = XSIZE*YSIZE;
int sizeTot = XSIZE*YSIZE;
float *kx = NULL;
cudaMalloc(&kx, XSIZE*YSIZE);
float *ky = NULL;
cudaMalloc(&ky, XSIZE*YSIZE);
float *kz = NULL;
cudaMalloc(&kz, XSIZE*YSIZE);
float *kz_is_imag = NULL;
cudaMalloc(&kz_is_imag, XSIZE*YSIZE);
float *kz_oil = NULL;
cudaMalloc(&kz_oil, XSIZE*YSIZE);
float *kz_oil_is_imag = NULL;
cudaMalloc(&kz_oil_is_imag, XSIZE*YSIZE);
float *pupil = NULL;
cudaMalloc(&pupil, XSIZE*YSIZE);
float *phase = NULL;
cudaMalloc(&phase, XSIZE*YSIZE);
float *position = NULL;
cudaMalloc(&position, XSIZE*YSIZE);
int *sparseIndexEvenDisk = NULL;
cudaMalloc(&sparseIndexEvenDisk, XSIZE*YSIZE);
int *sparseIndexOddDisk = NULL;
cudaMalloc(&sparseIndexOddDisk, XSIZE*YSIZE);
float *fft = NULL;
cudaMalloc(&fft, XSIZE*YSIZE);
int many = 1;
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
vec_computePSF_phaseNManywithOil_f<<<gridBlock,threadBlock>>>(n,sizePart,sizeTot,kx,ky,kz,kz_is_imag,kz_oil,kz_oil_is_imag,pupil,phase,position,sparseIndexEvenDisk,sparseIndexOddDisk,fft,many);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
vec_computePSF_phaseNManywithOil_f<<<gridBlock,threadBlock>>>(n,sizePart,sizeTot,kx,ky,kz,kz_is_imag,kz_oil,kz_oil_is_imag,pupil,phase,position,sparseIndexEvenDisk,sparseIndexOddDisk,fft,many);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
vec_computePSF_phaseNManywithOil_f<<<gridBlock,threadBlock>>>(n,sizePart,sizeTot,kx,ky,kz,kz_is_imag,kz_oil,kz_oil_is_imag,pupil,phase,position,sparseIndexEvenDisk,sparseIndexOddDisk,fft,many);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}