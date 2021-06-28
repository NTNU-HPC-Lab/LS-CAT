#include "includes.h"
__global__ void __ADAGrad(int nrows, int ncols, float *mm, float *um, float *ssq, float *mask, int maskr, float nw, float *ve, int nve, float *ts, int nts, float *lr, int nlr, float langevin, float eps, int doupdate, curandState *rstates) {
int ithread = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
int nthreads = blockDim.x * gridDim.x * gridDim.y;
int i, irow, icol;
float mmval, umval, sqrtss, sqrtnewss, veval, tsval, lrval, denom, grad;
float sqrtnw = sqrtf(nw);
float sqrt1mnw = sqrtf(1-nw);
float sqrteps = sqrt(eps);
curandState *prstate = &rstates[ithread];
for (i = ithread; i < nrows*ncols; i += nthreads) {
icol = i / nrows;
irow = i - icol * nrows;
umval = um[i];
sqrtss = ssq[i];
//    newsumsq = (nw * umval * umval) + (1 - nw) * sumsq;
sqrtnewss = hypotf(sqrtnw * umval, sqrt1mnw * sqrtss);
ssq[i] = sqrtnewss;
if (doupdate) {
mmval = mm[i];
veval = (nve > 1) ? ve[irow] : ve[0];
tsval = (nts > 1) ? ts[irow] : ts[0];
lrval = (nlr > 1) ? lr[irow] : lr[0];
sqrtnewss = hypotf(sqrtnewss, sqrteps);
denom = (veval == 0.5f) ? sqrtnewss : powf(sqrtnewss, veval*2);
grad = (umval / denom);
if (langevin > 0) grad += curand_normal(prstate) * langevin;
mmval += grad * lrval * tsval;
if (maskr > 0) {
if (maskr > 1) {
mmval *= mask[i];
} else {
mmval *= mask[icol];
}
}
mm[i] = mmval;
}
}
}