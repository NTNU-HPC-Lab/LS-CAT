#include "includes.h"
__global__ void cuSetupSincKernel_kernel(float *r_filter_, const int i_filtercoef_, const float r_soff_, const float r_wgthgt_, const int i_weight_, const float r_soff_inverse_, const float r_beta_, const float r_decfactor_inverse_, const float r_relfiltlen_inverse_)
{
int i = threadIdx.x + blockDim.x*blockIdx.x;
if(i > i_filtercoef_) return;
float r_wa = i - r_soff_;
float r_wgt = (1.0f - r_wgthgt_) + r_wgthgt_*cos(PI*r_wa*r_soff_inverse_);
float r_s = r_wa*r_beta_*r_decfactor_inverse_*PI;
float r_fct;
if(r_s != 0.0f) {
r_fct = sin(r_s)/r_s;
}
else {
r_fct = 1.0f;
}
if(i_weight_ == 1) {
r_filter_[i] = r_fct*r_wgt;
}
else {
r_filter_[i] = r_fct;
}
//printf("kernel %d %f\n", i, r_filter_[i]);
}