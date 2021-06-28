#include "includes.h"






__global__ void NmDistanceGradKernel(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
for (int i=blockIdx.x;i<b;i+=gridDim.x){
for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
float x1=xyz1[(i*n+j)*3+0];
float y1=xyz1[(i*n+j)*3+1];
float z1=xyz1[(i*n+j)*3+2];
int j2=idx1[i*n+j];
float x2=xyz2[(i*m+j2)*3+0];
float y2=xyz2[(i*m+j2)*3+1];
float z2=xyz2[(i*m+j2)*3+2];
float g=grad_dist1[i*n+j]*2;
atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*(x1-x2));
atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*(y1-y2));
atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*(z1-z2));
atomicAdd(&(grad_xyz2[(i*m+j2)*3+0]),-(g*(x1-x2)));
atomicAdd(&(grad_xyz2[(i*m+j2)*3+1]),-(g*(y1-y2)));
atomicAdd(&(grad_xyz2[(i*m+j2)*3+2]),-(g*(z1-z2)));
}
}
}