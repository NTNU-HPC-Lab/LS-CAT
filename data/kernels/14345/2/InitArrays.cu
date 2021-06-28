#include "includes.h"
__global__ void InitArrays(float *ip, float *op, float *fp, int *kp, int ncols)
{
int i;
float *fppos, *oppos, *ippos;
int *kppos;
int blockOffset;
int rowStartPos;
int colsPerThread;

// Each block gets a row, each thread will fill part of a row

// Calculate the offset of the row
blockOffset = blockIdx.x * ncols;
// Calculate our offset into the row
rowStartPos = threadIdx.x * (ncols/blockDim.x);
// The number of cols per thread
colsPerThread = ncols/blockDim.x;

ippos = ip + blockOffset+ rowStartPos;
fppos = fp + blockOffset+ rowStartPos;
oppos = op + blockOffset+ rowStartPos;
kppos = kp + blockOffset+ rowStartPos;

for (i = 0; i < colsPerThread; i++) {
fppos[i] = NOTSETLOC; // Not Fixed
ippos[i] = 50;
oppos[i] = 50;
kppos[i] = 1; // Keep Going
}
if(rowStartPos == 0) {
fppos[0] = SETLOC;
ippos[0] = 0;
oppos[0] = 0;
kppos[0] = 0;
}
if(rowStartPos + colsPerThread >= ncols) {
fppos[colsPerThread-1] = SETLOC;
ippos[colsPerThread-1] = 0;
oppos[colsPerThread-1] = 0;
kppos[colsPerThread-1] = 0;
}
if(blockOffset == 0) {
for(i=0;i < colsPerThread; i++) {
fppos[i] = SETLOC;
ippos[i] = 0;
oppos[i] = 0;
kppos[i] = 0;
}
}
if(blockOffset == ncols - 1) {
for(i=0;i < colsPerThread; i++) {
fppos[i] = SETLOC;
ippos[i] = 100;
oppos[i] = 100;
kppos[i] = 0;
}
}
if(blockOffset == 400 && rowStartPos < 330) {
if(rowStartPos + colsPerThread > 330) {
int end = 330 - rowStartPos;
for(i=0;i<end;i++) {
fppos[i] = SETLOC;
ippos[i] = 100;
oppos[i] = 100;
kppos[i] = 0;
}
}
else {
for(i=0;i<colsPerThread;i++) {
fppos[i] = SETLOC;
ippos[i] = 100;
oppos[i] = 100;
kppos[i] = 0;
}
}
}
if(blockOffset == 200 && rowStartPos <= 500 && rowStartPos + colsPerThread >=500) {
i=500-rowStartPos;
fppos[i] = SETLOC;
ippos[i] = 100;
oppos[i] = 100;
kppos[i] = 0;

}
// Insert code to set the rest of the boundary and fixed positions
}