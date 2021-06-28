#include "includes.h"
__global__ void CalculateFixed( const float *background, const float *target, const float *mask, float *fixed, const int wb, const int hb, const int wt, const int ht, const int oy, const int ox )
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

fixed[curt*3+0] = 4.0f*target[curt*3+0]-(target[North_t*3+0]+target[West_t*3+0]+target[South_t*3+0]+target[East_t*3+0]);
fixed[curt*3+1] = 4.0f*target[curt*3+1]-(target[North_t*3+1]+target[West_t*3+1]+target[South_t*3+1]+target[East_t*3+1]);
fixed[curt*3+2] = 4.0f*target[curt*3+2]-(target[North_t*3+2]+target[West_t*3+2]+target[South_t*3+2]+target[East_t*3+2]);

const int yb = oy+yt, xb = ox+xt;
const int curb = wb*yb+xb;
if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
bool nb_bnd = (yb == 0), wb_bnd = (xb == 0), sb_bnd = (yb == hb-1), eb_bnd = (xb == wb-1);
int North_b = (nb_bnd)? (curb):(curb-wb);
int West_b  = (wb_bnd)? (curb):(curb-1);
int South_b = (sb_bnd)? (curb):(curb+wb);
int East_b  = (eb_bnd)? (curb):(curb+1);

bool isMasked_n = (nt_bnd)? true:(mask[North_t] <= 127.0f);
bool isMasked_w = (wt_bnd)? true:(mask[West_t]  <= 127.0f);
bool isMasked_s = (st_bnd)? true:(mask[South_t] <= 127.0f);
bool isMasked_e = (et_bnd)? true:(mask[East_t]  <= 127.0f);

if(isMasked_n) {
fixed[curt*3+0] += background[North_b*3+0];
fixed[curt*3+1] += background[North_b*3+1];
fixed[curt*3+2] += background[North_b*3+2];
}

if(isMasked_w) {
fixed[curt*3+0] += background[West_b*3+0];
fixed[curt*3+1] += background[West_b*3+1];
fixed[curt*3+2] += background[West_b*3+2];
}

if(isMasked_s) {
fixed[curt*3+0] += background[South_b*3+0];
fixed[curt*3+1] += background[South_b*3+1];
fixed[curt*3+2] += background[South_b*3+2];
}

if(isMasked_e) {
fixed[curt*3+0] += background[East_b*3+0];
fixed[curt*3+1] += background[East_b*3+1];
fixed[curt*3+2] += background[East_b*3+2];
}
}
}
}