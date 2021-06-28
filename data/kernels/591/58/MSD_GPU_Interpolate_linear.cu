#include "includes.h"
__global__ void MSD_GPU_Interpolate_linear(float *d_MSD_DIT, float *d_MSD_interpolated, int *d_MSD_DIT_widths, int MSD_DIT_size, int *boxcar, int max_width_performed){

int tid  = threadIdx.x;
if(boxcar[tid] <= max_width_performed) {
//      int f = threadIdx.x;
int desired_width = boxcar[tid];
int position = (int) floorf(log2f((float) desired_width));

float width1 = d_MSD_DIT_widths[position];
float mean1 = d_MSD_DIT[(position)*MSD_RESULTS_SIZE];
float StDev1 = d_MSD_DIT[(position)*MSD_RESULTS_SIZE +1];

//      printf("\nBoxcar: %f \t desired: %f", (float)boxcar[f], desired_width);

if(position == MSD_DIT_size-1 && width1==(int) desired_width) {
//                (*mean) = mean1;
//                (*StDev) = StDev1;
d_MSD_interpolated[tid*2] = mean1;
d_MSD_interpolated[tid*2+1] = StDev1;
}
else {
float width2 = d_MSD_DIT_widths[position+1];
float distance_in_width = width2 - width1;

float mean2 = d_MSD_DIT[(position+1)*MSD_RESULTS_SIZE];
float distance_in_mean = mean2 - mean1;

float StDev2 = d_MSD_DIT[(position+1)*MSD_RESULTS_SIZE +1];
float distance_in_StDev = StDev2 - StDev1;

//                        printf("Position: \t %i \t f: %i\n", position, f);
//                        printf("width:[%f;%f]; mean:[%f;%f]; sd:[%f;%f]\n",width1, width2, mean1, mean2, StDev1, StDev2);
//                        printf("d width %f; d mean: %f; d StDef: %f\n", distance_in_width, distance_in_mean, distance_in_StDev);
//                        printf("\tDesired_width: %f\n", desired_width);

//                (*mean) = mean1 + (distance_in_mean/distance_in_width)*((float) desired_width - width1);
//                (*StDev) = StDev1 + (distance_in_StDev/distance_in_width)*((float) desired_width - width1);
d_MSD_interpolated[tid*2] = mean1 + (distance_in_mean/distance_in_width)*((float) desired_width - width1);
d_MSD_interpolated[tid*2+1] = StDev1 + (distance_in_StDev/distance_in_width)*((float) desired_width - width1);

}
}
}