#include "includes.h"
__device__ float sigmoid(float data){ return 1./(1. + expf(-data)); };
__global__ void yoloKernel(const int n,const float * input, float* output, const int* anchors,int anchor_num, int classes,int height,int width,float down_stride,float thresh){
const int idx = blockIdx.x * blockDim.x + threadIdx.x;
if(idx >= n) return;
extern __shared__ int shared_anchors[];
if(threadIdx.x < anchor_num*2){
shared_anchors[threadIdx.x] = anchors[threadIdx.x];
}
__syncthreads();
int row = idx % width;
int col = (idx / width) % height;
int anchor_id = (idx / width / height)% anchor_num;
int batch_id = idx/width/height/anchor_num;
int C = anchor_num*(classes+5);
int stride = width*height;
int begin_id =  ((batch_id * C + anchor_id*(classes + 5))*height+col)*width+row;
float conf_prob =sigmoid(input[begin_id + 4*stride]);
if(conf_prob > thresh) {
int class_id = -1;
float max_prob = thresh;
for (int c = 0;c<classes;++c){
int cls_id = begin_id + stride*(c + 5);
float cls_prob =  sigmoid(input[cls_id]) *conf_prob ;
if(cls_prob > max_prob){
max_prob = cls_prob;
class_id = c;
}
}
if(class_id >= 0){
int resCount = (int)atomicAdd(output,1);
float * data = output + 1 + resCount*7;
// x1,y1,x2,y2,cls,conf,batch_id
data[0] = (row + sigmoid(input[begin_id]))*down_stride;
data[1] = (col  + sigmoid(input[begin_id+stride]))*down_stride;
data[2] = expf(input[begin_id+2*stride]) * (float)shared_anchors[2*anchor_id];
data[3] = expf(input[begin_id+3*stride]) * (float)shared_anchors[2*anchor_id + 1];
data[4] = class_id;
data[5] = max_prob;
data[6] = batch_id;
}
}
}