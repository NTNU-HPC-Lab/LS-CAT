#include "includes.h"
__global__ void NmDistanceGradKernel(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
for (int i=blockIdx.x;i<b;i+=gridDim.x){
for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
float x1=xyz1[(i*n+j)*5+0];
float y1=xyz1[(i*n+j)*5+1];
float r1=xyz1[(i*n+j)*5+2];
float g1=xyz1[(i*n+j)*5+3];
float b1=xyz1[(i*n+j)*5+4];
int j2=idx1[i*n+j];
float x2=xyz2[(i*m+j2)*5+0];
float y2=xyz2[(i*m+j2)*5+1];
float r2=xyz2[(i*m+j2)*5+2];
float g2=xyz2[(i*m+j2)*5+3];
float b2=xyz2[(i*m+j2)*5+4];
float g=grad_dist1[i*n+j]*2;
atomicAdd(&(grad_xyz1[(i*n+j)*5+0]),g*(x1-x2));
atomicAdd(&(grad_xyz1[(i*n+j)*5+1]),g*(y1-y2));
atomicAdd(&(grad_xyz1[(i*n+j)*5+2]),g*(r1-r2));
atomicAdd(&(grad_xyz1[(i*n+j)*5+3]),g*(g1-g2));
atomicAdd(&(grad_xyz1[(i*n+j)*5+4]),g*(b1-b2));
atomicAdd(&(grad_xyz2[(i*m+j2)*5+0]),-(g*(x1-x2)));
atomicAdd(&(grad_xyz2[(i*m+j2)*5+1]),-(g*(y1-y2)));
atomicAdd(&(grad_xyz2[(i*m+j2)*5+2]),-(g*(r1-r2)));
atomicAdd(&(grad_xyz2[(i*m+j2)*5+3]),-(g*(g1-g2)));
atomicAdd(&(grad_xyz2[(i*m+j2)*5+4]),-(g*(b1-b2)));
}
}
}