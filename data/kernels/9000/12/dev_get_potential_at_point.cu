#include "includes.h"
__global__ void dev_get_potential_at_point( float eps2, float *eps, float *xh, float *yh, float *zh, float *xt, float *yt, float *zt, float *phi, int n, float *field_m, float *fxh, float *fyh, float *fzh, float *fxt, float *fyt, float *fzt, int n_field) {
float dx, dy, dz, r, dr2, eps2_total;
for (int tid=threadIdx.x + blockIdx.x*blockDim.x; tid < n; tid += blockDim.x*gridDim.x){
eps2_total = eps2 + eps[tid]*eps[tid];
phi[tid] = 0;
for (int i=0; i < n_field; i++){
dx = (fxh[i] - xh[tid]) + (fxt[i] - xt[tid]);
dy = (fyh[i] - yh[tid]) + (fyt[i] - yt[tid]);
dz = (fzh[i] - zh[tid]) + (fzt[i] - zt[tid]);
dr2 = dx*dx + dy*dy + dz*dz;
if (dr2 > 0) {
r = sqrt(eps2_total + dr2);
phi[tid] -= field_m[i] / r;
}
}
}
}