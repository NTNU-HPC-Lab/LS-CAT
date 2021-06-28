#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "cuSearchDoublet.cu"
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
const int *nSpM = NULL;
cudaMalloc(&nSpM, XSIZE*YSIZE);
const float *spMmat = NULL;
cudaMalloc(&spMmat, XSIZE*YSIZE);
const int *nSpB = NULL;
cudaMalloc(&nSpB, XSIZE*YSIZE);
const float *spBmat = NULL;
cudaMalloc(&spBmat, XSIZE*YSIZE);
const int *nSpT = NULL;
cudaMalloc(&nSpT, XSIZE*YSIZE);
const float *spTmat = NULL;
cudaMalloc(&spTmat, XSIZE*YSIZE);
const float *deltaRMin = NULL;
cudaMalloc(&deltaRMin, XSIZE*YSIZE);
const float *deltaRMax = NULL;
cudaMalloc(&deltaRMax, XSIZE*YSIZE);
const float *cotThetaMax = NULL;
cudaMalloc(&cotThetaMax, XSIZE*YSIZE);
const float *collisionRegionMin = NULL;
cudaMalloc(&collisionRegionMin, XSIZE*YSIZE);
const float *collisionRegionMax = NULL;
cudaMalloc(&collisionRegionMax, XSIZE*YSIZE);
int *nSpMcomp = NULL;
cudaMalloc(&nSpMcomp, XSIZE*YSIZE);
int *nSpBcompPerSpM_Max = NULL;
cudaMalloc(&nSpBcompPerSpM_Max, XSIZE*YSIZE);
int *nSpTcompPerSpM_Max = NULL;
cudaMalloc(&nSpTcompPerSpM_Max, XSIZE*YSIZE);
int *nSpBcompPerSpM = NULL;
cudaMalloc(&nSpBcompPerSpM, XSIZE*YSIZE);
int *nSpTcompPerSpM = NULL;
cudaMalloc(&nSpTcompPerSpM, XSIZE*YSIZE);
int *McompIndex = NULL;
cudaMalloc(&McompIndex, XSIZE*YSIZE);
int *BcompIndex = NULL;
cudaMalloc(&BcompIndex, XSIZE*YSIZE);
int *tmpBcompIndex = NULL;
cudaMalloc(&tmpBcompIndex, XSIZE*YSIZE);
int *TcompIndex = NULL;
cudaMalloc(&TcompIndex, XSIZE*YSIZE);
int *tmpTcompIndex = NULL;
cudaMalloc(&tmpTcompIndex, XSIZE*YSIZE);
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
cuSearchDoublet<<<gridBlock,threadBlock>>>(nSpM,spMmat,nSpB,spBmat,nSpT,spTmat,deltaRMin,deltaRMax,cotThetaMax,collisionRegionMin,collisionRegionMax,nSpMcomp,nSpBcompPerSpM_Max,nSpTcompPerSpM_Max,nSpBcompPerSpM,nSpTcompPerSpM,McompIndex,BcompIndex,tmpBcompIndex,TcompIndex,tmpTcompIndex);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
cuSearchDoublet<<<gridBlock,threadBlock>>>(nSpM,spMmat,nSpB,spBmat,nSpT,spTmat,deltaRMin,deltaRMax,cotThetaMax,collisionRegionMin,collisionRegionMax,nSpMcomp,nSpBcompPerSpM_Max,nSpTcompPerSpM_Max,nSpBcompPerSpM,nSpTcompPerSpM,McompIndex,BcompIndex,tmpBcompIndex,TcompIndex,tmpTcompIndex);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
cuSearchDoublet<<<gridBlock,threadBlock>>>(nSpM,spMmat,nSpB,spBmat,nSpT,spTmat,deltaRMin,deltaRMax,cotThetaMax,collisionRegionMin,collisionRegionMax,nSpMcomp,nSpBcompPerSpM_Max,nSpTcompPerSpM_Max,nSpBcompPerSpM,nSpTcompPerSpM,McompIndex,BcompIndex,tmpBcompIndex,TcompIndex,tmpTcompIndex);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}