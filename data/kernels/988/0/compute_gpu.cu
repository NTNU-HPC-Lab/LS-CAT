#include "includes.h"
#define K 3
#define BLCH 8
#define BLCW 32

__global__ void compute_gpu(float *img, float *f, float * out, int bh, int bw, int imgH, int imgW, int imgN, int nF, int convH, int convW){
int idY = blockDim.y * blockIdx.y + threadIdx.y;
int idX = blockDim.x * blockIdx.x + threadIdx.x;

int inm1, inm2, inm3, inm4, inf, ind1, ind2, ind3;
inm1 = 0;
inf = 0;
ind1 = 0;

for (int mi = 0; mi < imgN; mi++){
ind1 += convW * convH;
inm1 += imgW * imgH;
if (idX < convH && idY < convW){
ind2 = ind1 + convW * idX;
inm2 = inm1 + imgW * idX;
ind3 = ind2 + idY;
inm3 = inm2 + idY;
for (int fi = 0; fi < nF; fi++){
inm4 = inm3 + imgW * fi;
inf = ind3*nF*nF + fi*nF;
for (int fj = 0; fj < nF; fj++){
out[ind3] += img[inm4+fj] * f[inf+fj];
}
}
}
}
}