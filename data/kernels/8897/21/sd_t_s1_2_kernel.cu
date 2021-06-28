#include "includes.h"
__global__ void sd_t_s1_2_kernel(size_t h1d,size_t h2d,size_t h3d,size_t p4d,size_t p6d,size_t p4ld_t2,size_t h1ld_t2,size_t h3ld_v2,size_t h2ld_v2,size_t p6ld_v2,size_t h3ld_t3,size_t h2ld_t3,size_t h1ld_t3,size_t p6ld_t3,size_t p4ld_t3,double *t2_d, double *v2_d,size_t p4, size_t total_x, double* t3d) {
size_t h1,h2,h3,p6;
__shared__ double t2_shm[T1*4*Tcomm];

for(size_t i=threadIdx.x;i<h1d*p4d;i+=blockDim.x)
if(i<h1d*p4d)
t2_shm[i] = t2_d[i];
size_t rest_x=blockIdx.x;
size_t thread_x = T2*T1 * rest_x + threadIdx.x;
rest_x = thread_x;
__syncthreads();
/* the following computation may need to happen inside the loop */
for(size_t i=0;i<total_x;i+=gridDim.x*blockDim.x)
{
rest_x += i;
h3=rest_x%h3d;
rest_x=rest_x/h3d;
h2=rest_x%h2d;
rest_x=rest_x/h2d;
p6=rest_x%p6d;

if((thread_x+i)<total_x)
for(h1=0;h1<h1d;h1++)
for(p4=0;p4<p4d;p4++)
{
t3d[h3*h3ld_t3+h2*h2ld_t3+h1*h1ld_t3+p6*p6ld_t3+p4*p4ld_t3]-=t2_shm[h1*p4d+p4]*v2_d[h3*h3ld_v2+h2*h2ld_v2+p6*p6ld_v2];
}
}
__syncthreads();
}