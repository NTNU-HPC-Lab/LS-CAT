#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "updateGradInputLSM.cu"
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
const float *target = NULL;
cudaMalloc(&target, XSIZE*YSIZE);
const float *mapping = NULL;
cudaMalloc(&mapping, XSIZE*YSIZE);
const float *n_class_in_cluster = NULL;
cudaMalloc(&n_class_in_cluster, XSIZE*YSIZE);
float *class_score = NULL;
cudaMalloc(&class_score, XSIZE*YSIZE);
float *class_logsum = NULL;
cudaMalloc(&class_logsum, XSIZE*YSIZE);
float *cluster_score = NULL;
cudaMalloc(&cluster_score, XSIZE*YSIZE);
float *cluster_logsum = NULL;
cudaMalloc(&cluster_logsum, XSIZE*YSIZE);
const long class_score_stride0 = 1;
const long cluster_score_stride0 = 1;
int n_clusters = 1;
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
updateGradInputLSM<<<gridBlock,threadBlock>>>(target,mapping,n_class_in_cluster,class_score,class_logsum,cluster_score,cluster_logsum,class_score_stride0,cluster_score_stride0,n_clusters);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
updateGradInputLSM<<<gridBlock,threadBlock>>>(target,mapping,n_class_in_cluster,class_score,class_logsum,cluster_score,cluster_logsum,class_score_stride0,cluster_score_stride0,n_clusters);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
updateGradInputLSM<<<gridBlock,threadBlock>>>(target,mapping,n_class_in_cluster,class_score,class_logsum,cluster_score,cluster_logsum,class_score_stride0,cluster_score_stride0,n_clusters);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}