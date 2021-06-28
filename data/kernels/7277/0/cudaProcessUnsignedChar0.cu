#include "includes.h"
__global__ void cudaProcessUnsignedChar0(unsigned char *dst, unsigned char *src, int imgW, int imgH)
{
int tx = threadIdx.x;
int ty = threadIdx.y;
int bw = blockDim.x;
int bh = blockDim.y;
int x = blockIdx.x*bw + tx * 2;
int y = blockIdx.y*bh + ty * 2;
int px = y * imgW + x;

bool flag = 0 < y && y < (imgH - 2) && 0 < x && x < (imgW - 2);
int sx1 = flag ? px - imgW : 0;
int sx2 = flag ? px - imgW + 1 : 0;
int sx3 = flag ? px - imgW + 2 : 0;
int sx4 = flag ? px - 1 : 0;
int sx5 = flag ? px : 0;
int sx6 = flag ? px + 1 : 0;
int sx7 = flag ? px + 2 : 0;
int sx8 = flag ? px + imgW - 1 : 0;
int sx9 = flag ? px + imgW : 0;
int sxa = flag ? px + imgW + 1 : 0;
int sxb = flag ? px + imgW + 2 : 0;
int sxc = flag ? px + imgW * 2 - 1 : 0;
int sxd = flag ? px + imgW * 2     : 0;
int sxe = flag ? px + imgW * 2 + 1 : 0;

// G0 R0 G1 R1    x0 x1 x2 x3
// B0 G2 B1 G3    x4 x5 x6 x7
// G4 R2 G5 R3    x8 x9 xA xB
// B2 G6 B3 G7    xC xD xE xF

int g1 = (int)src[sx2];
int g2 = (int)src[sx5];
int g3 = (int)src[sx7];
int g4 = (int)src[sx8];
int g5 = (int)src[sxa];
int g6 = (int)src[sxd];
int b0 = (int)src[sx4];
int b1 = (int)src[sx6];
int b2 = (int)src[sxc];
int b3 = (int)src[sxe];
int r0 = (int)src[sx1];
int r1 = (int)src[sx3];
int r2 = (int)src[sx9];
int r3 = (int)src[sxb];

int db0 = (b0 + b1) >> 1;
int dg0 = g2;
int dr0 = (r0 + r1) >> 1;
int db1 = b1;
int dg1 = (g1 + g2 + g3 + g5) >> 2;
int dr1 = (r0 + r1 + r2 + r3) >> 2;
int db2 = (b0 + b1 + b2 + b3) >> 2;
int dg2 = (g2 + g4 + g5 + g6) >> 2;
int dr2 = r2;
int db3 = (b1 + b3) >> 1;
int dg3 = g5;
int dr3 = (r2 + r3) >> 1;

int dx = px * 3;
int dst0 = dx;
int dst1 = dx + 3;
int dst2 = dx + imgW * 3;
int dst3 = dx + (imgW + 1) * 3;
dst[dst0 + 0 < imgW * imgH * 3 ? dst0 + 0 : 0] = (unsigned char)db0;
dst[dst0 + 1 < imgW * imgH * 3 ? dst0 + 1 : 0] = (unsigned char)dg0;
dst[dst0 + 2 < imgW * imgH * 3 ? dst0 + 2 : 0] = (unsigned char)dr0;
dst[dst1 + 0 < imgW * imgH * 3 ? dst1 + 0 : 0] = (unsigned char)db1;
dst[dst1 + 1 < imgW * imgH * 3 ? dst1 + 1 : 0] = (unsigned char)dg1;
dst[dst1 + 2 < imgW * imgH * 3 ? dst1 + 2 : 0] = (unsigned char)dr1;
dst[dst2 + 0 < imgW * imgH * 3 ? dst2 + 0 : 0] = (unsigned char)db2;
dst[dst2 + 1 < imgW * imgH * 3 ? dst2 + 1 : 0] = (unsigned char)dg2;
dst[dst2 + 2 < imgW * imgH * 3 ? dst2 + 2 : 0] = (unsigned char)dr2;
dst[dst3 + 0 < imgW * imgH * 3 ? dst3 + 0 : 0] = (unsigned char)db3;
dst[dst3 + 1 < imgW * imgH * 3 ? dst3 + 1 : 0] = (unsigned char)dg3;
dst[dst3 + 2 < imgW * imgH * 3 ? dst3 + 2 : 0] = (unsigned char)dr3;
}