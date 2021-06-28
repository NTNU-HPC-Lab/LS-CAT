#include "includes.h"
__global__ void calculate_IMC(float *norm,float *IMC,float *HX,float *HY,float *entropy,float *px,float *py,float *HXY,int max,float sum,int size){
//printf("%d\n",max);
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy * max + ix;
int tid=threadIdx.x;

int i;
for(i=0;i<max;i++){
if(idx>=i*max && idx<(i+1)*(max) && norm[idx]>0){
HX[idx]=-(norm[idx]*log10f(norm[idx]));
//printf("%d,i %d  %f %f \n",idx,i,miu_x[idx],norm[idx]);
}
}

if(idx<size && norm[idx] !=0){
entropy[idx]=-(norm[idx]*log10f(norm[idx]));
//printf("%d f3 %f \n",idx,entropy[idx]);
__syncthreads();
}



// for(i=0;i<max;i++){
//     if(idx>=i*max && idx<(i+1)*(max) && norm[idx]>0){
//         px[idx]=norm[idx];
//         //printf("%d,i %d  %f %f \n",idx,i,miu_x[idx],norm[idx]);
// }
// }
if(idx<size){
px[idx]=norm[idx];
}

int c=0;
for(i=0;i<max;i++){
// printf("%d",batas);
if(c==i && idx<max){
py[c*max+idx]=norm[idx*max+i];
//printf("%d %d,i %d  %f %f %d \n",idx,idx,i,stdy[idx],norm[idx*max+i],idx*max+i);
c++;
}

//printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
}


int b=0;
for(i=0;i<max;i++){
// printf("%d",batas);
if(b==i && idx<max &&norm[idx*max+i]>0){
HY[b*max+idx]=-(norm[idx*max+i]*log10f(norm[idx*max+i]));
//printf("%d %d,i %d  %f %f %d \n",idx,idx,i,HY[b*max+idx],norm[idx*max+i],b*max+i);
b++;
}

//printf("xx %d %f\n",idx*i+idx,miu_x[idx]);
}




for (int stride = 1; stride < size; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{
HX[idx] += HX[idx+ stride];
HY[idx] += HY[idx+ stride];
px[idx] += px[idx+ stride];
py[idx] += py[idx+ stride];
entropy[idx] += entropy[idx+ stride];
}
// synchronize within threadblock
__syncthreads();
}


if(idx>9000){
HXY[idx]=abs(norm[idx]*(log10f((px[0]*py[0]))));
//printf("tid %d %f %f %f %f \n",idx,HXY[idx],px[0],py[0],norm[idx]);
}

for (int stride = 1; stride < size; stride *= 2)
{
if ((tid % (2 * stride)) == 0)
{
HXY[idx] += HXY[idx+ stride];


}
// synchronize within threadblock
__syncthreads();
}

if (idx == 0){
if(HX[0]>HY[0]){
IMC[0]=(entropy[0]-HXY[0])/HX[0];
//printf("x%f %f %f %f px%f %f\n",abs(IMC[0]),entropy[0],HXY[0],HX[0],px[0],py[0]);
}
else{
IMC[0]=entropy[0]-HXY[0]/HY[0];
//printf("y%f %f %f %f\n",abs(IMC[0]),entropy[0],HXY[0],HY[0]);
}
printf("IMC %f\n",abs(IMC[0]));
}
}