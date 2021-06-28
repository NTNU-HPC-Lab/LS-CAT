#include "includes.h"
__global__ void __fillToInds4DLong(long long A, long long *B, int ldb, int rdb, int tdb, int *I, int nrows, int *J, int ncols, int *K, int nk, int *L, int nl) {
int tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
int step = blockDim.x * gridDim.x * gridDim.y;
int l = tid / (nrows * ncols * nk);
int tidrem = tid - l * (nrows * ncols * nk);
int lstep = step / (nrows * ncols * nk);
int steprem = step - lstep * (nrows * ncols * nk);
int k = tidrem / (nrows * ncols);
tidrem = tidrem - k * (nrows * ncols);
int kstep = steprem / (nrows * ncols);
steprem = steprem - kstep * (nrows * ncols);
int j = tidrem / nrows;
int i = tidrem - j * nrows;
int jstep = steprem / nrows;
int istep = steprem - jstep * nrows;
int id, mapi, mapj, mapk, mapl;
for (id = tid; id < nrows * ncols * nk * nl; id += step) {
mapl = l;
if (L != NULL) mapl = L[l];
mapk = k;
if (K != NULL) mapk = K[k];
mapj = j;
if (J != NULL) mapj = J[j];
mapi = i;
if (I != NULL) mapi = I[i];
B[mapi + ldb * (mapj + rdb * (mapk + tdb * mapl))] = A;
i += istep;
if (i >= nrows) {i -= nrows; j++;}
j += jstep;
if (j >= ncols) {j -= ncols; k++;}
k += kstep;
if (k >= nk) {k -= nk; l++;}
l += lstep;
}
}