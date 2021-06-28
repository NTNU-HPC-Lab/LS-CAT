#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "cudaSNormalizeROIs_kernel.cu"
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
unsigned int inputSizeX = 1;
unsigned int inputSizeY = 1;
unsigned int nbProposals = 1;
unsigned int batchSize = 1;
unsigned int scoreIdx = 1;
unsigned int nbCls = 1;
unsigned int maxParts = 1;
unsigned int maxTemplates = 1;
bool keepMax = 1;
bool generateParts = 1;
bool generateTemplates = 1;
const float normX = 1;
const float normY = 1;
const float *means = NULL;
cudaMalloc(&means, XSIZE*YSIZE);
const float *std = NULL;
cudaMalloc(&std, XSIZE*YSIZE);
const unsigned int *numPartsPerClass = NULL;
cudaMalloc(&numPartsPerClass, XSIZE*YSIZE);
const unsigned int *numTemplatesPerClass = NULL;
cudaMalloc(&numTemplatesPerClass, XSIZE*YSIZE);
const float *ROIRef = NULL;
cudaMalloc(&ROIRef, XSIZE*YSIZE);
const float *ROIEst = NULL;
cudaMalloc(&ROIEst, XSIZE*YSIZE);
const float *ValuesEst = NULL;
cudaMalloc(&ValuesEst, XSIZE*YSIZE);
const float *partsEst = NULL;
cudaMalloc(&partsEst, XSIZE*YSIZE);
const float *partsVisibilityEst = NULL;
cudaMalloc(&partsVisibilityEst, XSIZE*YSIZE);
const float *templatesEst = NULL;
cudaMalloc(&templatesEst, XSIZE*YSIZE);
float *outputs = NULL;
cudaMalloc(&outputs, XSIZE*YSIZE);
int *argMax = NULL;
cudaMalloc(&argMax, XSIZE*YSIZE);
float *partsPrediction = NULL;
cudaMalloc(&partsPrediction, XSIZE*YSIZE);
float *partsVisibilityPrediction = NULL;
cudaMalloc(&partsVisibilityPrediction, XSIZE*YSIZE);
float *templatesPrediction = NULL;
cudaMalloc(&templatesPrediction, XSIZE*YSIZE);
float scoreThreshold = 1;
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
cudaSNormalizeROIs_kernel<<<gridBlock,threadBlock>>>(inputSizeX,inputSizeY,nbProposals,batchSize,scoreIdx,nbCls,maxParts,maxTemplates,keepMax,generateParts,generateTemplates,normX,normY,means,std,numPartsPerClass,numTemplatesPerClass,ROIRef,ROIEst,ValuesEst,partsEst,partsVisibilityEst,templatesEst,outputs,argMax,partsPrediction,partsVisibilityPrediction,templatesPrediction,scoreThreshold);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
cudaSNormalizeROIs_kernel<<<gridBlock,threadBlock>>>(inputSizeX,inputSizeY,nbProposals,batchSize,scoreIdx,nbCls,maxParts,maxTemplates,keepMax,generateParts,generateTemplates,normX,normY,means,std,numPartsPerClass,numTemplatesPerClass,ROIRef,ROIEst,ValuesEst,partsEst,partsVisibilityEst,templatesEst,outputs,argMax,partsPrediction,partsVisibilityPrediction,templatesPrediction,scoreThreshold);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
cudaSNormalizeROIs_kernel<<<gridBlock,threadBlock>>>(inputSizeX,inputSizeY,nbProposals,batchSize,scoreIdx,nbCls,maxParts,maxTemplates,keepMax,generateParts,generateTemplates,normX,normY,means,std,numPartsPerClass,numTemplatesPerClass,ROIRef,ROIEst,ValuesEst,partsEst,partsVisibilityEst,templatesEst,outputs,argMax,partsPrediction,partsVisibilityPrediction,templatesPrediction,scoreThreshold);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}