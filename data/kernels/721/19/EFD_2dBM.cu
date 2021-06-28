#include "includes.h"
__global__ void EFD_2dBM( int width, int height, int pitch_n, int pitch_npo, float *d_val_n, float *d_val_npo, float alpha, float beta ){
int idx = blockIdx.x;	//row
int idy = threadIdx.x;	//column

if ((idx < height) && (idy <width )){
//d_val_npo[i] = Pu * d_val_n[i + 1] + Pm * d_val_n[i] + Pd * d_val_n[i - 1];
d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = alpha*(d_val_n[(idx+1)*(pitch_n / sizeof(float)) + idy]
+ d_val_n[(idx - 1)*(pitch_n / sizeof(float)) + idy])
+ beta*(d_val_n[idx*(pitch_n / sizeof(float)) + idy+1]
+ d_val_n[idx*(pitch_n / sizeof(float)) + idy-1])
+ (1.0-2.0*alpha-2.0*beta)*d_val_n[idx*(pitch_n / sizeof(float)) + idy];

//modify the ones on the top
if (idx == 0){
d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = d_val_npo[(idx+1)*(pitch_npo / sizeof(float)) + idy];
}
//modify the ones on the bottom
if (idx == (height-1)){
d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = d_val_npo[(idx - 1)*(pitch_npo / sizeof(float)) + idy];
}
//modify the ones on the left
if (idy == 0){
d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = d_val_npo[(idx - 1)*(pitch_npo / sizeof(float)) + idy+1];
}
//modify the ones on the right
if (idx == (width - 1)){
d_val_npo[idx*(pitch_npo / sizeof(float)) + idy] = d_val_npo[(idx - 1)*(pitch_npo / sizeof(float)) + idy-1];
}
}
}