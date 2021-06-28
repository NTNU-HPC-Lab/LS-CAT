#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "forwardPathKernel.cu"
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
int noPaths = 1;
int nYears = 1;
int noSpecies = 1;
int noPatches = 1;
int noControls = 1;
int noUncertainties = 1;
float timeStep = 1;
float *initPops = NULL;
cudaMalloc(&initPops, XSIZE*YSIZE);
float *pops = NULL;
cudaMalloc(&pops, XSIZE*YSIZE);
float *mmm = NULL;
cudaMalloc(&mmm, XSIZE*YSIZE);
int *rowIdx = NULL;
cudaMalloc(&rowIdx, XSIZE*YSIZE);
int *elemsPerCol = NULL;
cudaMalloc(&elemsPerCol, XSIZE*YSIZE);
int maxElems = 1;
float *speciesParams = NULL;
cudaMalloc(&speciesParams, XSIZE*YSIZE);
float *caps = NULL;
cudaMalloc(&caps, XSIZE*YSIZE);
float *aars = NULL;
cudaMalloc(&aars, XSIZE*YSIZE);
float *uncertParams = NULL;
cudaMalloc(&uncertParams, XSIZE*YSIZE);
int *controls = NULL;
cudaMalloc(&controls, XSIZE*YSIZE);
float *uJumps = NULL;
cudaMalloc(&uJumps, XSIZE*YSIZE);
float *uBrownian = NULL;
cudaMalloc(&uBrownian, XSIZE*YSIZE);
float *uJumpSizes = NULL;
cudaMalloc(&uJumpSizes, XSIZE*YSIZE);
float *uJumpsSpecies = NULL;
cudaMalloc(&uJumpsSpecies, XSIZE*YSIZE);
float *uBrownianSpecies = NULL;
cudaMalloc(&uBrownianSpecies, XSIZE*YSIZE);
float *uJumpSizesSpecies = NULL;
cudaMalloc(&uJumpSizesSpecies, XSIZE*YSIZE);
float *rgr = NULL;
cudaMalloc(&rgr, XSIZE*YSIZE);
float *uResults = NULL;
cudaMalloc(&uResults, XSIZE*YSIZE);
float *totalPops = NULL;
cudaMalloc(&totalPops, XSIZE*YSIZE);
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
forwardPathKernel<<<gridBlock,threadBlock>>>(noPaths,nYears,noSpecies,noPatches,noControls,noUncertainties,timeStep,initPops,pops,mmm,rowIdx,elemsPerCol,maxElems,speciesParams,caps,aars,uncertParams,controls,uJumps,uBrownian,uJumpSizes,uJumpsSpecies,uBrownianSpecies,uJumpSizesSpecies,rgr,uResults,totalPops);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
forwardPathKernel<<<gridBlock,threadBlock>>>(noPaths,nYears,noSpecies,noPatches,noControls,noUncertainties,timeStep,initPops,pops,mmm,rowIdx,elemsPerCol,maxElems,speciesParams,caps,aars,uncertParams,controls,uJumps,uBrownian,uJumpSizes,uJumpsSpecies,uBrownianSpecies,uJumpSizesSpecies,rgr,uResults,totalPops);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
forwardPathKernel<<<gridBlock,threadBlock>>>(noPaths,nYears,noSpecies,noPatches,noControls,noUncertainties,timeStep,initPops,pops,mmm,rowIdx,elemsPerCol,maxElems,speciesParams,caps,aars,uncertParams,controls,uJumps,uBrownian,uJumpSizes,uJumpsSpecies,uBrownianSpecies,uJumpSizesSpecies,rgr,uResults,totalPops);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}