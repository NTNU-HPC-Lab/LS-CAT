#include "includes.h"
__global__ void __fillToInds3D(double A, double *B, int ldb, int rdb, int *I, int nrows, int *J, int ncols, int *K, int nk) {
int ii = threadIdx.x + blockDim.x * blockIdx.x;
int jj = threadIdx.y + blockDim.y * blockIdx.y;
int kk = threadIdx.z + blockDim.z * blockIdx.z;
int i, j, k, mapi, mapj, mapk;
for (k = kk; k < nk; k += blockDim.z * gridDim.z) {
mapk = k;
if (K != NULL) mapk = K[k];
for (j = jj; j < ncols; j += blockDim.y * gridDim.y) {
mapj = j;
if (J != NULL) mapj = J[j];
if (I != NULL) {
for (i = ii; i < nrows; i += blockDim.x * gridDim.x) {
mapi = I[i];
B[mapi + ldb * (mapj + rdb * mapk)] = A;
}
} else {
for (i = ii; i < nrows; i += blockDim.x * gridDim.x) {
mapi = i;
B[mapi + ldb * (mapj + rdb * mapk)] = A;
}
}
}
}
}