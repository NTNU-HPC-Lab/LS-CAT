#include "includes.h"
__device__ int is_source_gpu(int i, int j, int radius, int source_active, int src_x, int src_y)
{
if (!source_active)
return 0;
if (sqrt(pow((float)(src_x - i), 2) + pow((float)(src_y - j), 2)) <= radius)
return 1;
return 0;
}
__global__ void wireless_src_pulse_kernel(int step, double amp, double MAX_TIME, double TIME_STEP, int radius, int source_active, int src_x, int src_y, double *ua_gpu, double *ub_gpu, double *uc_gpu)
{
int i, j;
int i_start, j_start;
int i_final, j_final;
int line_length;
int global_thread_x, global_thread_y;
int thread_work = 32;

line_length = gridDim.y * blockDim.y;

global_thread_x = blockDim.x * blockIdx.x + threadIdx.x;
global_thread_y = blockDim.y * blockIdx.y + threadIdx.y;

i_start	= global_thread_x * thread_work;
j_start	= global_thread_y * thread_work;
i_final	= global_thread_x * (thread_work + 1);
j_final = global_thread_y * (thread_work + 1);

if (step < (int)(MAX_TIME / TIME_STEP) / 2){
// Pulse source
for (i = i_start; i < i_final; i++){
for (j = j_start; j < j_final; j++){
if (is_source_gpu(i, j, radius, 1, src_x, src_y))
uc_gpu[i * line_length + j] = amp * fabs(sin(step * M_PI/4));
}
}
} else if (source_active){
for (i = i_start; i < i_final; i++) {
for (j = j_start; j < j_final; j++) {
if (is_source_gpu(i, j, radius, source_active, src_x, src_y)) {
ua_gpu[i * line_length + j] = 0;
ub_gpu[i * line_length + j] = 0;
uc_gpu[i * line_length + j] = 0;
}
}
}
}
// All threads should reach this point before setting source_active.
// Option 1:  need a thread barrier here -> not done, I chose option 2
// Option 2:  simply write 2 kernels and syncCPU -> done, I chose this option
//	 	CPU is setting source_active = 0 after this kernel is done executing.
}