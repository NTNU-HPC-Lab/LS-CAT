#include "includes.h"
__global__ void FindPos(int *pos, bool *forest, int text_size, int order, int step)
{
int text_idx = blockIdx.x * blockDim.x + threadIdx.x;
int offset = blockIdx.x*step;
if(text_idx < text_size) {
if(!forest[offset+blockDim.x+threadIdx.x]) {
pos[text_idx] = 0;
} else {
bool isCurBlock = true;
bool isLeftMost = (blockIdx.x < 1);
int  nodeIdx    = blockDim.x+threadIdx.x;
int  leftBound  = blockDim.x;
int  rightBound = 2*blockDim.x-1;
int  alignOrder = 0;
// bottom-up
while(alignOrder != order) {
int leftInx;
if(nodeIdx-1 < leftBound) {
if(isLeftMost) break;
isCurBlock = false;
leftInx = offset-step+rightBound;
} else {
leftInx = offset+nodeIdx-1;
}

if(!forest[leftInx]) break;

rightBound = leftBound-1;
leftBound /= 2;
nodeIdx /= 2;
alignOrder++;
}

// top-down
if(alignOrder == order && !isLeftMost) isCurBlock = false;
nodeIdx = (!isCurBlock)? rightBound
:(nodeIdx-1 < leftBound)? nodeIdx
:nodeIdx-1;

offset = offset - ((isCurBlock)? 0:step);
while(alignOrder != 0) {
if((alignOrder == order && isCurBlock) || forest[offset+2*nodeIdx+1]) {
nodeIdx = 2*nodeIdx;
} else {
nodeIdx = 2*nodeIdx+1;
}
alignOrder--;
}

pos[text_idx] = (isCurBlock)? (threadIdx.x-(nodeIdx-blockDim.x)+(forest[offset+nodeIdx]))
:(step-nodeIdx+threadIdx.x);
}
}
}