#include "includes.h"
__global__ void update(double *weights_in_d, double *weights_h_d, double *weights_out_d, double *weights_in_delta_d, double *weights_h_delta_d, double *weights_out_delta_d, double *error_d){
int tix = threadIdx.x;

if(tix < INPUTS*H_HEIGHT){
weights_in_d[tix] -= (alpha_d * weights_in_delta_d[tix] / 55);
weights_in_delta_d[tix] = 0.0;
}

weights_h_d[tix] -= (alpha_d * weights_h_delta_d[tix] / 55);
weights_h_delta_d[tix] = 0.0;

if(tix < OUTPUTS*H_HEIGHT){
weights_out_d[tix] -= (alpha_d * weights_out_delta_d[tix] / 55);
weights_out_delta_d[tix] = 0.0;
}

if(tix < 1){
error_d[0] = error_d[0] * 100.0 / 55;
printf("\nGPU Error: %f\n", error_d[0]);
error_d[0] = 0;
}

}