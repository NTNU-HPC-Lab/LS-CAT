#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#include "filter_kernel.cu"
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
const float *box_preds = NULL;
cudaMalloc(&box_preds, XSIZE*YSIZE);
const float *cls_preds = NULL;
cudaMalloc(&cls_preds, XSIZE*YSIZE);
const float *dir_preds = NULL;
cudaMalloc(&dir_preds, XSIZE*YSIZE);
const int *anchor_mask = NULL;
cudaMalloc(&anchor_mask, XSIZE*YSIZE);
const float *dev_anchors_px = NULL;
cudaMalloc(&dev_anchors_px, XSIZE*YSIZE);
const float *dev_anchors_py = NULL;
cudaMalloc(&dev_anchors_py, XSIZE*YSIZE);
const float *dev_anchors_pz = NULL;
cudaMalloc(&dev_anchors_pz, XSIZE*YSIZE);
const float *dev_anchors_dx = NULL;
cudaMalloc(&dev_anchors_dx, XSIZE*YSIZE);
const float *dev_anchors_dy = NULL;
cudaMalloc(&dev_anchors_dy, XSIZE*YSIZE);
const float *dev_anchors_dz = NULL;
cudaMalloc(&dev_anchors_dz, XSIZE*YSIZE);
const float *dev_anchors_ro = NULL;
cudaMalloc(&dev_anchors_ro, XSIZE*YSIZE);
float *filtered_box = NULL;
cudaMalloc(&filtered_box, XSIZE*YSIZE);
float *filtered_score = NULL;
cudaMalloc(&filtered_score, XSIZE*YSIZE);
int *filtered_dir = NULL;
cudaMalloc(&filtered_dir, XSIZE*YSIZE);
float *box_for_nms = NULL;
cudaMalloc(&box_for_nms, XSIZE*YSIZE);
int *filter_count = NULL;
cudaMalloc(&filter_count, XSIZE*YSIZE);
const float FLOAT_MIN = 1;
const float FLOAT_MAX = 1;
const float score_threshold = 1;
const int NUM_BOX_CORNERS = 1;
const int NUM_OUTPUT_BOX_FEATURE = 1;
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
filter_kernel<<<gridBlock,threadBlock>>>(box_preds,cls_preds,dir_preds,anchor_mask,dev_anchors_px,dev_anchors_py,dev_anchors_pz,dev_anchors_dx,dev_anchors_dy,dev_anchors_dz,dev_anchors_ro,filtered_box,filtered_score,filtered_dir,box_for_nms,filter_count,FLOAT_MIN,FLOAT_MAX,score_threshold,NUM_BOX_CORNERS,NUM_OUTPUT_BOX_FEATURE);
cudaDeviceSynchronize();
for (int loop_counter = 0; loop_counter < 10; ++loop_counter) {
filter_kernel<<<gridBlock,threadBlock>>>(box_preds,cls_preds,dir_preds,anchor_mask,dev_anchors_px,dev_anchors_py,dev_anchors_pz,dev_anchors_dx,dev_anchors_dy,dev_anchors_dz,dev_anchors_ro,filtered_box,filtered_score,filtered_dir,box_for_nms,filter_count,FLOAT_MIN,FLOAT_MAX,score_threshold,NUM_BOX_CORNERS,NUM_OUTPUT_BOX_FEATURE);
}
auto start = steady_clock::now();
for (int loop_counter = 0; loop_counter < 1000; loop_counter++) {
filter_kernel<<<gridBlock,threadBlock>>>(box_preds,cls_preds,dir_preds,anchor_mask,dev_anchors_px,dev_anchors_py,dev_anchors_pz,dev_anchors_dx,dev_anchors_dy,dev_anchors_dz,dev_anchors_ro,filtered_box,filtered_score,filtered_dir,box_for_nms,filter_count,FLOAT_MIN,FLOAT_MAX,score_threshold,NUM_BOX_CORNERS,NUM_OUTPUT_BOX_FEATURE);
}
auto end = steady_clock::now();
auto usecs = duration_cast<duration<float, microseconds::period> >(end - start);
cout <<'['<<usecs.count()<<','<<'('<<BLOCKX<<','<<BLOCKY<<')' << ','<<'('<<XSIZE<<','<<YSIZE<<')'<<']' << endl;
}
}}