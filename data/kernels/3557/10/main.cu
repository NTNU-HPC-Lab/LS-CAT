#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "cudaS_ssdToOutput_kernels.cu"
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
unsigned int batchSize = 1;
unsigned int nbClass = 1;
unsigned int nbAnchors = 1;
unsigned int channelWidth = 1;
unsigned int channelHeight = 1;
unsigned int nbProposals = 1;
unsigned int *nbValidROIs = NULL;
cudaMalloc(&nbValidROIs, XSIZE*YSIZE);
unsigned int cls = 1;
unsigned int totalParts = 1;
unsigned int totalTemplates = 1;
unsigned int maxParts = 1;
unsigned int maxTemplates = 1;
unsigned int cumulParts = 1;
unsigned int cumulTemplates = 1;
unsigned int nbParts = 1;
unsigned int nbTemplates = 1;
float xRatio = 1;
float yRatio = 1;
float xOutputRatio = 1;
float yOutputRatio = 1;
const float *roi_bbox = NULL;
cudaMalloc(&roi_bbox, XSIZE*YSIZE);
const float *roi_anchors = NULL;
cudaMalloc(&roi_anchors, XSIZE*YSIZE);
const float *anchors = NULL;
cudaMalloc(&anchors, XSIZE*YSIZE);
const float *inputs_parts = NULL;
cudaMalloc(&inputs_parts, XSIZE*YSIZE);
const float *inputs_templates = NULL;
cudaMalloc(&inputs_templates, XSIZE*YSIZE);
float *outputs = NULL;
cudaMalloc(&outputs, XSIZE*YSIZE);
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
cudaS_ssdToOutput_kernels<<<gridBlock,threadBlock>>>(batchSize,nbClass,nbAnchors,channelWidth,channelHeight,nbProposals,nbValidROIs,cls,totalParts,totalTemplates,maxParts,maxTemplates,cumulParts,cumulTemplates,nbParts,nbTemplates,xRatio,yRatio,xOutputRatio,yOutputRatio,roi_bbox,roi_anchors,anchors,inputs_parts,inputs_templates,outputs);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
cudaS_ssdToOutput_kernels<<<gridBlock,threadBlock>>>(batchSize,nbClass,nbAnchors,channelWidth,channelHeight,nbProposals,nbValidROIs,cls,totalParts,totalTemplates,maxParts,maxTemplates,cumulParts,cumulTemplates,nbParts,nbTemplates,xRatio,yRatio,xOutputRatio,yOutputRatio,roi_bbox,roi_anchors,anchors,inputs_parts,inputs_templates,outputs);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
cudaS_ssdToOutput_kernels<<<gridBlock,threadBlock>>>(batchSize,nbClass,nbAnchors,channelWidth,channelHeight,nbProposals,nbValidROIs,cls,totalParts,totalTemplates,maxParts,maxTemplates,cumulParts,cumulTemplates,nbParts,nbTemplates,xRatio,yRatio,xOutputRatio,yOutputRatio,roi_bbox,roi_anchors,anchors,inputs_parts,inputs_templates,outputs);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}