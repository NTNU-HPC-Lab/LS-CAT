#include "includes.h"
__global__ void countQueens(int* frontQueensPos, int* data, int* numFQP)
{
int localResult = 0;
//printf("%d\n", numFQP[0]);
int thisThread = ((blockIdx.x * gridDim.x + blockIdx.y) * gridDim.y + threadIdx.x)* blockDim.x + threadIdx.y;
//	printf("1_%d %d %d %d %d %d %d %d\n", thisThread, blockIdx.x, gridDim.x, blockIdx.y, gridDim.y, threadIdx.x, blockDim.x, threadIdx.y);
//	if (thisThread >= QUEENS * QUEENS * QUEENS * QUEENS)
//		return;
if (blockIdx.x >= QUEENS || blockIdx.y >= QUEENS || threadIdx.x >= QUEENS || threadIdx.y >= QUEENS)
return;

int* queenPos = new int[QUEENS];

queenPos[3] = blockIdx.x;
queenPos[4] = blockIdx.y;
queenPos[5] = threadIdx.x;
queenPos[6] = threadIdx.y;

for (int i = 4; i <= 6; i++) {
for (int j = 3; j < i; j++) {
if ((queenPos[i] - i) == (queenPos[j] - j) || (queenPos[i] + i) == (queenPos[j] + j) || queenPos[i] == queenPos[j]) {
return;
}
}
}
int totalFQP = numFQP[0] / 3;

for (int FQP_number = 0; FQP_number < totalFQP; FQP_number++) {
//	printf("1_%d %d %d %d %d %d %d %d\n", thisThread, blockIdx.x, gridDim.x, blockIdx.y, gridDim.y, threadIdx.x, blockDim.x, threadIdx.y);
//	if (thisThread >= QUEENS * QUEENS * QUEENS * QUEENS)
//		return;

for (int i = 0; i < 3; i++)
queenPos[i] = frontQueensPos[(FQP_number * 3) + i];

bool legal = true;

//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
//	printf("1_%d %d %d %d %d %d %d_%d\n", queenPos[0], queenPos[1], queenPos[2], queenPos[3], queenPos[4], queenPos[5], queenPos[6], totalFQP);

for (int i = 3; i <= 6; i++) {
for (int j = 0; j < 3; j++) {
if ((queenPos[i] - i) == (queenPos[j] - j) || (queenPos[i] + i) == (queenPos[j] + j) || queenPos[i] == queenPos[j]) {
legal = false;
break;
}
}
if (!legal)
break;
}
if (!legal)
continue;

//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
//	printf("1_%d %d %d %d %d %d %d_%d\n", queenPos[0], queenPos[1], queenPos[2], queenPos[3], queenPos[4], queenPos[5], queenPos[6], localResult);

//printf("1_%d %d %d %d %d %d %d\n", thisThread, queenPos[2], blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, data[thisThread]);
//backtrace
int posNow = 7;
queenPos[posNow] = -1;
while (posNow > 6) {
queenPos[posNow] ++;
while (queenPos[posNow] < QUEENS) {
legal = true;
for (int j = posNow - 1; j >= 0; j--) {
if ((queenPos[posNow] - posNow) == (queenPos[j] - j) || (queenPos[posNow] + posNow) == (queenPos[j] + j) || queenPos[posNow] == queenPos[j]) {
legal = false;
break;
}
}
if (!legal)
queenPos[posNow] ++;
else
break;
}
if (queenPos[posNow] < QUEENS) {
if (posNow == (QUEENS - 1)) {
localResult++;
//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
//	printf("2_%d %d %d %d %d %d %d_%d\n", queenPos[7], queenPos[8], queenPos[9], queenPos[10], queenPos[11], queenPos[12], queenPos[13], localResult);
posNow--;
}
else {
posNow++;
queenPos[posNow] = -1;
}
}
else
posNow--;
}
}
//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
//	printf("2.5_%d\n", localResult);
data[thisThread] = localResult;
//if (blockIdx.x == 6 && blockIdx.y == 11 && threadIdx.x == 9 && threadIdx.y == 12)
//	printf("3_%d %d %d %d %d %d\n", thisThread, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, data[thisThread]);
}