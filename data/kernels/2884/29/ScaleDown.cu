#include "includes.h"
__global__ void ScaleDown(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
__shared__ float inrow[SCALEDOWN_W+4];
__shared__ float brow[5*(SCALEDOWN_W/2)];
__shared__ int yRead[SCALEDOWN_H+4];
__shared__ int yWrite[SCALEDOWN_H+4];
#define dx2 (SCALEDOWN_W/2)
const int tx = threadIdx.x;
const int tx0 = tx + 0*dx2;
const int tx1 = tx + 1*dx2;
const int tx2 = tx + 2*dx2;
const int tx3 = tx + 3*dx2;
const int tx4 = tx + 4*dx2;
const int xStart = blockIdx.x*SCALEDOWN_W;
const int yStart = blockIdx.y*SCALEDOWN_H;
const int xWrite = xStart/2 + tx;
float k0 = d_ScaleDownKernel[0];
float k1 = d_ScaleDownKernel[1];
float k2 = d_ScaleDownKernel[2];
if (tx<SCALEDOWN_H+4) {
int y = yStart + tx - 2;
y = (y<0 ? 0 : y);
y = (y>=height ? height-1 : y);
yRead[tx] = y*pitch;
yWrite[tx] = (yStart + tx - 4)/2 * newpitch;
}
__syncthreads();
int xRead = xStart + tx - 2;
xRead = (xRead<0 ? 0 : xRead);
xRead = (xRead>=width ? width-1 : xRead);

int maxtx = min(dx2, width/2 - xStart/2);
for (int dy=0;dy<SCALEDOWN_H+4;dy+=5) {
{
inrow[tx] = d_Data[yRead[dy+0] + xRead];
__syncthreads();
if (tx<maxtx) {
brow[tx4] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
if (dy>=4 && !(dy&1))
d_Result[yWrite[dy+0] + xWrite] = k2*brow[tx2] + k0*(brow[tx0]+brow[tx4]) + k1*(brow[tx1]+brow[tx3]);
}
__syncthreads();
}
if (dy<(SCALEDOWN_H+3)) {
inrow[tx] = d_Data[yRead[dy+1] + xRead];
__syncthreads();
if (tx<maxtx) {
brow[tx0] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
if (dy>=3 && (dy&1))
d_Result[yWrite[dy+1] + xWrite] = k2*brow[tx3] + k0*(brow[tx1]+brow[tx0]) + k1*(brow[tx2]+brow[tx4]);
}
__syncthreads();
}
if (dy<(SCALEDOWN_H+2)) {
inrow[tx] = d_Data[yRead[dy+2] + xRead];
__syncthreads();
if (tx<maxtx) {
brow[tx1] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
if (dy>=2 && !(dy&1))
d_Result[yWrite[dy+2] + xWrite] = k2*brow[tx4] + k0*(brow[tx2]+brow[tx1]) + k1*(brow[tx3]+brow[tx0]);
}
__syncthreads();
}
if (dy<(SCALEDOWN_H+1)) {
inrow[tx] = d_Data[yRead[dy+3] + xRead];
__syncthreads();
if (tx<maxtx) {
brow[tx2] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
if (dy>=1 && (dy&1))
d_Result[yWrite[dy+3] + xWrite] = k2*brow[tx0] + k0*(brow[tx3]+brow[tx2]) + k1*(brow[tx4]+brow[tx1]);
}
__syncthreads();
}
if (dy<SCALEDOWN_H) {
inrow[tx] = d_Data[yRead[dy+4] + xRead];
__syncthreads();
if (tx<dx2 && xWrite<width/2) {
brow[tx3] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
if (!(dy&1))
d_Result[yWrite[dy+4] + xWrite] = k2*brow[tx1] + k0*(brow[tx4]+brow[tx3]) + k1*(brow[tx0]+brow[tx2]);
}
__syncthreads();
}
}
}