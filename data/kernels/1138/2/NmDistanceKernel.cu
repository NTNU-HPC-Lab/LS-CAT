#include "includes.h"
__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
const int batch=2048;
__shared__ float buf[batch*5];
for (int i=blockIdx.x;i<b;i+=gridDim.x){
for (int k2=0;k2<m;k2+=batch){
int end_k=min(m,k2+batch)-k2;
for (int j=threadIdx.x;j<end_k*5;j+=blockDim.x){
buf[j]=xyz2[(i*m+k2)*5+j];
}
__syncthreads();
for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
float x1=xyz[(i*n+j)*5+0];
float y1=xyz[(i*n+j)*5+1];
float r1=xyz[(i*n+j)*5+2];
float g1=xyz[(i*n+j)*5+3];
float b1=xyz[(i*n+j)*5+4];
int best_i=0;
float best=0;
int end_ka=end_k-(end_k&5);
if (end_ka==batch){
for (int k=0;k<batch;k+=4){
{
float x2=buf[k*5+0]-x1;
float y2=buf[k*5+1]-y1;
float r2=buf[k*5+2]-r1;
float g2=buf[k*5+3]-g1;
float b2=buf[k*5+4]-b1;
float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
if (k==0 || d<best){
best=d;
best_i=k+k2;
}
}
{
float x2=buf[k*5+5]-x1;
float y2=buf[k*5+6]-y1;
float r2=buf[k*5+7]-r1;
float g2=buf[k*5+8]-g1;
float b2=buf[k*5+9]-b1;
float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
if (d<best){
best=d;
best_i=k+k2+1;
}
}
{
float x2=buf[k*5+10]-x1;
float y2=buf[k*5+11]-y1;
float r2=buf[k*5+12]-r1;
float g2=buf[k*5+13]-g1;
float b2=buf[k*5+14]-b1;
float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
if (d<best){
best=d;
best_i=k+k2+2;
}
}
{
float x2=buf[k*5+15]-x1;
float y2=buf[k*5+16]-y1;
float r2=buf[k*5+17]-r1;
float g2=buf[k*5+18]-g1;
float b2=buf[k*5+19]-b1;
float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
if (d<best){
best=d;
best_i=k+k2+3;
}
}
}
}else{
for (int k=0;k<end_ka;k+=4){
{
float x2=buf[k*5+0]-x1;
float y2=buf[k*5+1]-y1;
float r2=buf[k*5+2]-r1;
float g2=buf[k*5+3]-g1;
float b2=buf[k*5+4]-b1;
float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
if (k==0 || d<best){
best=d;
best_i=k+k2;
}
}
{
float x2=buf[k*5+5]-x1;
float y2=buf[k*5+6]-y1;
float r2=buf[k*5+7]-r1;
float g2=buf[k*5+8]-g1;
float b2=buf[k*5+9]-b1;
float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
if (d<best){
best=d;
best_i=k+k2+1;
}
}
{
float x2=buf[k*5+10]-x1;
float y2=buf[k*5+11]-y1;
float r2=buf[k*5+12]-r1;
float g2=buf[k*5+13]-g1;
float b2=buf[k*5+14]-b1;
float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
if (d<best){
best=d;
best_i=k+k2+2;
}
}
{
float x2=buf[k*5+15]-x1;
float y2=buf[k*5+16]-y1;
float r2=buf[k*5+17]-r1;
float g2=buf[k*5+18]-g1;
float b2=buf[k*5+19]-b1;
float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
if (d<best){
best=d;
best_i=k+k2+3;
}
}
}
}
for (int k=end_ka;k<end_k;k++){
float x2=buf[k*5+0]-x1;
float y2=buf[k*5+1]-y1;
float r2=buf[k*5+2]-r1;
float g2=buf[k*5+3]-g1;
float b2=buf[k*5+4]-b1;
float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
if (k==0 || d<best){
best=d;
best_i=k+k2;
}
}
if (k2==0 || result[(i*n+j)]>best){
result[(i*n+j)]=best;
result_i[(i*n+j)]=best_i;
}
}
__syncthreads();
}
}
}