#include "includes.h"
__global__ void __fillToInds3DLong(long long A, long long *B, int ldb, int rdb, int *I, int nrows, int *J, int ncols, int *K, int nk) {
int tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
int step = blockDim.x * gridDim.x * gridDim.y;
int k = tid / (nrows * ncols);
int tidrem = tid - k * (nrows * ncols);
int kstep = step / (nrows * ncols);
int steprem = step - kstep * (nrows * ncols);
int j = tidrem / nrows;
int i = tidrem - j * nrows;
int jstep = steprem / nrows;
int istep = steprem - jstep * nrows;
int id, mapi, mapj, mapk;
for (id = tid; id < nrows * ncols * nk; id += step) {
mapk = k;
if (K != NULL) mapk = K[k];
mapj = j;
if (J != NULL) mapj = J[j];
mapi = i;
if (I != NULL) mapi = I[i];
B[mapi + ldb * (mapj + rdb * mapk)] = A;
i += istep;
if (i >= nrows) {i -= nrows; j++;}
j += jstep;
if (j >= ncols) {j -= ncols; k++;}
k += kstep;
}
}