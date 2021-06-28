#include "includes.h"
__global__ void _segmentedScanBackKer(float *maxdist, int *maxdistidx, int *label, float *blockmaxdist, int *blocklabel, int *blockmaxdistidx, int numelements)
{
// 声明共享内存。用来存放中间结果小数组中的元素，也就是输入的原数组的每块最
// 后一个元素。共包含三个信息。
__shared__ float shdcurmaxdist[1];
__shared__ int shdcurlabel[1];
__shared__ int shdcurmaxdistindex[1];


// 状态位，用来标记上一块的最后一个元素的标签值是否和本段第一个元素的标签值
// 相同。
__shared__ int state[1];

// 计算需要进行块间累加位置索引（块外的数组索引）。
int idx = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

// 用每块的第一个线程来读取每块前一块的最后一个元素，从中间结果数组中读取。
if (threadIdx.x == 0) {
shdcurmaxdist[0] = blockmaxdist[blockIdx.x];
shdcurlabel[0] = blocklabel[blockIdx.x];
shdcurmaxdistindex[0] = blockmaxdistidx[blockIdx.x];
// 用 state 来记录上一块的最后一个元素的标签值是否和本段第一个元素的
// 标签值相同，相同则为 1，不同则为 0。
state[0] = (label[idx] == shdcurlabel[0]);
}

// 块内同步。
__syncthreads();

// 如果状态位为 0，说明上一块和本块无关，不在一个区域内，直接返回。
if (state[0] == 0)
return;
// 如果数组索引大于数组长度，直接返回。
if (idx >= numelements)
return;
// 如果当前位置处的标签值和目前已知的最大垂距的标签值相同，并且垂距小于目前
// 已知的最大垂距，那么更新当前位置处的最大垂距记录和最大垂距位置的索引。
if (label[idx] == shdcurlabel[0] && maxdist[idx] < shdcurmaxdist[0]) {
maxdist[idx] = shdcurmaxdist[0];
maxdistidx[idx] = shdcurmaxdistindex[0];
}
}