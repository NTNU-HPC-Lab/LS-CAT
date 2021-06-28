#include "includes.h"

unsigned char *pdata; // pointer to data content

__global__ void processData(unsigned char *Da, int* filter)
{
int tx = threadIdx.x;           // thread的x軸id
int bx = blockIdx.x;            // block的x軸id
int bn = blockDim.x;
int gid = bx * bn + tx;
__shared__ int sfilter[3][3];
__shared__ int sR[3][512];      // 每個block存上中下三行
__shared__ int sG[3][512];
__shared__ int sB[3][512];
__shared__ int sRsum[512];      // 每個block 最後512個sum
__shared__ int sGsum[512];
__shared__ int sBsum[512];

if (tx < 9)                     // 每個block 存filter 到 share memory
{
sfilter[tx / 3][tx % 3] = filter[tx];
}
__syncthreads();

if (bx == 0 || bx == 511 || tx == 0 || tx == 511)
{
// 邊界處理 --> 直接給原本值不動
sRsum[tx] = Da[gid * 3];
sGsum[tx] = Da[gid * 3 + 1];
sBsum[tx] = Da[gid * 3 + 2];
}

// 邊界處理(第1個block跟最後一個block不做)
if (bx != 0 && bx != 511)
{
// R, G, B個別將該Row(Block)運算會用到的上中下三行存入Share Memory
sR[0][tx] = Da[gid * 3 - 512 * 3];
sR[1][tx] = Da[gid * 3];
sR[2][tx] = Da[gid * 3 + 512 * 3];

sG[0][tx] = Da[gid * 3 - 512 * 3 + 1];
sG[1][tx] = Da[gid * 3 + 1];
sG[2][tx] = Da[gid * 3 + 512 * 3 + 1];

sB[0][tx] = Da[gid * 3 - 512 * 3 + 2];
sB[1][tx] = Da[gid * 3 + 2];
sB[2][tx] = Da[gid * 3 + 512 * 3 + 2];
__syncthreads();

// 邊界處理(每個block的的第一個值和最後一個值不做)
if (tx != 0 && tx != 511)
{
// R
sRsum[tx] = sR[0][tx - 1] * sfilter[0][0];
sRsum[tx] += sR[0][tx] * sfilter[0][1];
sRsum[tx] += sR[0][tx + 1] * sfilter[0][2];

sRsum[tx] += sR[1][tx - 1] * sfilter[1][0];
sRsum[tx] += sR[1][tx] * sfilter[1][1];
sRsum[tx] += sR[1][tx + 1] * sfilter[1][2];

sRsum[tx] += sR[2][tx - 1] * sfilter[2][0];
sRsum[tx] += sR[2][tx] * sfilter[2][1];
sRsum[tx] += sR[2][tx + 1] * sfilter[2][2];

// G
sGsum[tx] = sG[0][tx - 1] * sfilter[0][0];
sGsum[tx] += sG[0][tx] * sfilter[0][1];
sGsum[tx] += sG[0][tx + 1] * sfilter[0][2];

sGsum[tx] += sG[1][tx - 1] * sfilter[1][0];
sGsum[tx] += sG[1][tx] * sfilter[1][1];
sGsum[tx] += sG[1][tx + 1] * sfilter[1][2];

sGsum[tx] += sG[2][tx - 1] * sfilter[2][0];
sGsum[tx] += sG[2][tx] * sfilter[2][1];
sGsum[tx] += sG[2][tx + 1] * sfilter[2][2];

// B
sBsum[tx] = sB[0][tx - 1] * sfilter[0][0];
sBsum[tx] += sB[0][tx] * sfilter[0][1];
sBsum[tx] += sB[0][tx + 1] * sfilter[0][2];

sBsum[tx] += sB[1][tx - 1] * sfilter[1][0];
sBsum[tx] += sB[1][tx] * sfilter[1][1];
sBsum[tx] += sB[1][tx + 1] * sfilter[1][2];

sBsum[tx] += sB[2][tx - 1] * sfilter[2][0];
sBsum[tx] += sB[2][tx] * sfilter[2][1];
sBsum[tx] += sB[2][tx + 1] * sfilter[2][2];


sRsum[tx] /= filter[9];
sGsum[tx] /= filter[9];
sBsum[tx] /= filter[9];
// 大於255 或 小於0處理
if (sRsum[tx] > 255)
sRsum[tx] = 255;
else if (sRsum[tx] < 0)
sRsum[tx] = 0;

if (sGsum[tx] > 255)
sGsum[tx] = 255;
else if (sGsum[tx] < 0)
sGsum[tx] = 0;

if (sBsum[tx] > 255)
sBsum[tx] = 255;
else if (sBsum[tx] < 0)
sBsum[tx] = 0;
}
}

__syncthreads();

// 將R, G, B三個陣列值合併寫回一維陣列，以利輸出到檔案
Da[gid * 3] = sRsum[tx];
Da[gid * 3 + 1] = sGsum[tx];
Da[gid * 3 + 2] = sBsum[tx];
}