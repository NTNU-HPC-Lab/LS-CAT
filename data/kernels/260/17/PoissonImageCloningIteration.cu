#include "includes.h"
__global__ void PoissonImageCloningIteration( const float *fixed, const float *mask, const float *buf1, float *buf2, const int wt, const int ht )
{
const int yt = blockIdx.y * blockDim.y + threadIdx.y;
const int xt = blockIdx.x * blockDim.x + threadIdx.x;
const int curt = wt*yt+xt;
if (yt < ht and xt < wt and mask[curt] > 127.0f) {
bool nt_bnd = (yt == 0), wt_bnd = (xt == 0), st_bnd = (yt == ht-1), et_bnd = (xt == wt-1);
int North_t = (nt_bnd)? curt:(curt-wt);
int West_t  = (wt_bnd)? curt:(curt-1);
int South_t = (st_bnd)? curt:(curt+wt);
int East_t  = (et_bnd)? curt:(curt+1);

bool isMasked_n = (nt_bnd)? true:(mask[North_t] <= 127.0f);
bool isMasked_w = (wt_bnd)? true:(mask[West_t]  <= 127.0f);
bool isMasked_s = (st_bnd)? true:(mask[South_t] <= 127.0f);
bool isMasked_e = (et_bnd)? true:(mask[East_t]  <= 127.0f);

buf2[curt*3+0] = fixed[curt*3+0];
buf2[curt*3+1] = fixed[curt*3+1];
buf2[curt*3+2] = fixed[curt*3+2];

if(!isMasked_n) {
buf2[curt*3+0] += buf1[North_t*3+0];
buf2[curt*3+1] += buf1[North_t*3+1];
buf2[curt*3+2] += buf1[North_t*3+2];
}

if(!isMasked_w) {
buf2[curt*3+0] += buf1[West_t*3+0];
buf2[curt*3+1] += buf1[West_t*3+1];
buf2[curt*3+2] += buf1[West_t*3+2];
}

if(!isMasked_s) {
buf2[curt*3+0] += buf1[South_t*3+0];
buf2[curt*3+1] += buf1[South_t*3+1];
buf2[curt*3+2] += buf1[South_t*3+2];
}

if(!isMasked_e) {
buf2[curt*3+0] += buf1[East_t*3+0];
buf2[curt*3+1] += buf1[East_t*3+1];
buf2[curt*3+2] += buf1[East_t*3+2];
}

buf2[curt*3+0] *= 0.25f;
buf2[curt*3+1] *= 0.25f;
buf2[curt*3+2] *= 0.25f;
}
}