#include "includes.h"
__global__ void vecAdd(float * in1, int offset, int len) {
//@@ Insert code to implement vector addition here
int i =  threadIdx.x;

if( (offset + i) <len ) in1[offset + i] = in1[offset + i]+in1[offset-1];
if( (offset + i + blockDim.x ) <len ) in1[offset + i+ blockDim.x] = in1[offset + i+ blockDim.x]+in1[offset-1];
}