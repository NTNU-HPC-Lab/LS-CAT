#include "includes.h"
__global__ void fsc_tomo_cmp_kernal(const float* data1, const float* data2, float* device_soln, const float data1threshold, const float data2threshold, const int nx, const int ny, const int nz, const int offset)
{

const uint x=threadIdx.x;
const uint y=blockIdx.x;

int idx = x + y*MAX_THREADS + offset;

float sum_data1_amps = 0.0;
float sum_data2_amps = 0.0;
float top = 0.0;
for(int i = 0; i < ny; i++){
//int index = i*nx + idx % nx + ((idx/nx)*ny*nz); //for coalesing
int rindex = i*nx + 2*(idx % nx/2) + (2*idx/nx)*ny*nz;
int iindex = i*nx + 2*(idx % nx/2)+ 1 + (2*idx/nx)*ny*nz;
float data1_r = data1[rindex];
float data1_i = data1[iindex];
float data2_r = data2[rindex];
float data2_i = data2[iindex];
if((data1_r* data1_r +  data1_i*data1_i) > data1threshold && (data2_r* data2_r +  data2_i*data2_i) > data2threshold){
sum_data1_amps += (data1_r* data1_r +  data1_i*data1_i);
sum_data2_amps += (data2_r* data2_r +  data2_i*data2_i);
top += (data1_r*data2_r + data1_i*data2_i);
}
}
device_soln[idx*3] = top;
device_soln[idx*3 +1] = sum_data1_amps;
device_soln[idx*3 +2] = sum_data2_amps;

}