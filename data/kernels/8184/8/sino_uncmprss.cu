#include "includes.h"
__global__ void sino_uncmprss(unsigned int * dsino, unsigned char * p1sino, unsigned char * d1sino, int ifrm, int nele)
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
if (idx<nele) {
d1sino[2 * idx] = (unsigned char)((dsino[ifrm*nele + idx] >> 8) & 0x000000ff);
d1sino[2 * idx + 1] = (unsigned char)((dsino[ifrm*nele + idx] >> 24) & 0x000000ff);

p1sino[2 * idx] = (unsigned char)(dsino[ifrm*nele + idx] & 0x000000ff);
p1sino[2 * idx + 1] = (unsigned char)((dsino[ifrm*nele + idx] >> 16) & 0x000000ff);
}
}