#include "includes.h"
__global__ void calculate_correlation(float *norm,float *corelation,float *miu_x,float *miu_y,float *stdx,float *stdy,int *ikj,float *dif_variance,int max,float sum,int size){
//printf("%d\n",max);
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * max + ix;
int tid=threadIdx.x;
int i;
for(i=0;i<max;i++){
if(idx>=i*max && idx<(i+1)*(max)){
miu_x[idx]=i*norm[idx];
//printf("%d,i %d  %f %f \n",idx,i,miu_x[idx],norm[idx]);
}

//printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
}
int blok=0;
for(i=0;i<max;i++){
if(blok==i && idx<max){
miu_y[blok*max+idx]=i*norm[idx*max+i];
//printf("%d %d,i %d  %f %f %d \n",idx,idx,i,miu_y[idx],norm[idx*max+i],idx*max+i);
blok++;
}

//printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
}
for(i=0;i<max;i++){
if(idx>=i*max && idx<(i+1)*(max)){
stdx[idx]=((i-miu_x[0])*(i-miu_x[0]))*norm[idx];
//printf("%d,i %d  %f %f \n",idx,i,miu_x[idx],norm[idx]);
}

//printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
}
int batas=0;
for(i=0;i<max;i++){
// printf("%d",batas);
if(batas==i && idx<max){
stdy[batas*max+idx]=((i-miu_y[0])*(i-miu_y[0]))*norm[idx*max+i];
//printf("%d %d,i %d  %f %f %d \n",idx,idx,i,stdy[idx],norm[idx*max+i],idx*max+i);
batas++;
}

//printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
}
if(idx==0){
for(i=0;i<max;i++){
for(int j=0;j<max;j++){
ikj[max*i+j]=i*j;
//printf("tid %d %d\n",max*i+j,ikj[max*i+j]);
}
}
}
if(idx<size){
corelation[idx]=((ikj[idx]*norm[idx]));
//printf("%d %d,i %d  %f %f \n",idx,idx,i,corelation[idx],norm[idx]);
}
for (int stride = 1; stride < size; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{
corelation[idx] += corelation[idx+ stride];
//printf("%d %f\n",idx,corelation[idx]);
}
// synchronize within threadblock
__syncthreads();
}
for (int stride = 1; stride < size; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{

miu_x[idx] += miu_x[idx+ stride];
stdy[idx] += stdy[idx+ stride];
miu_y[idx] += miu_y[idx+ stride];
stdx[idx] += stdx[idx+ stride];
// corelation[idx] += corelation[idx+ stride];
//printf("%d %f\n",idx,miu_x[idx]);
}
// synchronize within threadblock
__syncthreads();
}
int k=0;
if(idx==0){
for(i=0;i<max;i++){
for(int j=0;j<max;j++){
k=abs(i-j);
dif_variance[k]=((k-((miu_x[0]+miu_y[0])/2))*(k-((miu_x[0]+miu_y[0])/2)))*norm[k];

if(k=i){
dif_variance[k]+=dif_variance[i];
//printf("%d %f %f %f \n",k,dif_variance[k],(k-((miu_x[0]+miu_y[0])/2)),norm[k]);

}
}
}

}

for (int stride = 1; stride < size; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{
dif_variance[idx] +=dif_variance[idx+stride];
}
// synchronize within threadblock
__syncthreads();
}
if (idx == 0){

printf("correlation %f\n",abs(corelation[0]-miu_x[0]*miu_y[0])/stdx[0]*stdy[0]);
printf("variance %f\n",stdx[0]);
printf("difference variance %f\n",dif_variance[0]);
}
}