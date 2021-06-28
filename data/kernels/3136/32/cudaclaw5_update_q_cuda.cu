#include "includes.h"
__global__ void cudaclaw5_update_q_cuda(int mbc, double dtdx, double dtdy, double* qold, double* fm, double* fp, double* gm, double* gp)
{
int mq = threadIdx.z;
int x = threadIdx.x;
int x_stride = blockDim.z;
int y = threadIdx.y;
int y_stride = (blockDim.x + 2*mbc)*x_stride;
int i = mq + (x+mbc)*x_stride + (y+mbc)*y_stride;
qold[i] = qold[i] - dtdx * (fm[i+x_stride] - fp[i])
- dtdy * (gm[i+y_stride] - gp[i]);
}