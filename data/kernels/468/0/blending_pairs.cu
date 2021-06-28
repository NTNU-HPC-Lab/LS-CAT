#include "includes.h"
#define num_thread 256
#define num_block 256
__global__ void blending_pairs(float *a,float *b,float *c,float *d,float *wei,int width,int height,int w,float A,float error_lm,float error_mm,int class_num)
{
const int tid=threadIdx.x;
const int bid=blockIdx.x;
const int Idx=num_thread*bid+tid;
float r_LM,r_MM, r_center_LM,r_center_MM;
int row,column;
int i,j;
float sum1,sum2;
float st=0.0;
int judge;
float dis;
//float wei;
float weih,result;
int kk=0;
int rmin,rmax,smin,smax;
for(int kkk=Idx;kkk<width*height;kkk=kkk+num_thread*num_block)
{
result=0;
judge=0;
wei[kkk]=0;
kk=0;
sum1=0,sum2=0;
row=kkk/width;
column=kkk%width;
//if(row==1)
//	wei=0;
r_center_LM =d[kkk]-b[kkk]+error_lm;
r_center_MM=d[kkk]-c[kkk]+1.412*error_mm;
if(column-w/2<=0)
rmin=0;
else
rmin = column-w/2;

if(column+w/2>=width-1)
rmax = width-1;
else
rmax = column+w/2;

if(row-w/2<=0)
smin=0;
else
smin = row-w/2;

if(row+w/2>=height-1)
smax = height-1;
else
smax = row+w/2;
for(i=smin;i<=smax;i++)
{
for(j=rmin;j<=rmax;j++)
{
sum1+=b[i*width+j]*b[i*width+j];
sum2+=b[i*width+j];
}
}
//if(column==30&&row==30)
//	result=0;
st=sqrt(sum1/(w*w)-(sum2/(w*w))*(sum2/(w*w)))/ class_num;
for(i=smin;i<=smax;i++)
{
for(j=rmin;j<=rmax;j++)
{
if(fabs(b[kkk]-b[i*width+j])<st)
{
r_LM=d[i*width+j]-b[i*width+j];
r_MM=d[i*width+j]-c[i*width+j];
if((r_center_LM>0&&r_LM<r_center_LM)||(r_center_LM<0&&r_LM>r_center_LM))
{
if((r_center_MM>0&&r_MM<r_center_MM)||(r_center_MM<0&&r_MM>r_center_MM))
{
r_LM=fabs(r_LM)+0.0001;
r_MM=fabs(r_MM)+0.0001;
if(kkk==i*width+j)
judge=1;
dis=float((row-i)*(row-i)+(column-j)*(column-j));
dis=sqrt(dis)/A+1.0;
weih=1.0/(dis* r_LM*r_MM);
wei[kkk]+=weih;
result+=weih*(c[i*width+j]+b[i*width+j]-d[i*width+j]);
kk++;
}
}
}
}
}
if(kk==0)
{
a[kkk]=abs(b[kkk]+c[kkk]-d[kkk])*1000;
wei[kkk]=1000;

}
else
{
if(judge==0)
{
dis=1.0;
r_LM=fabs(d[kkk]-b[kkk])+0.0001;
r_MM=fabs(d[kkk]-c[kkk])+0.0001;
weih=1.0/(dis* r_LM*r_MM);
result+=weih*(b[kkk]+c[kkk]-d[kkk]);
wei[kkk]+=weih;
}
a[kkk]=result;
//if(a[kkk]<0)
//	a[kkk]=(b[kkk]+c[kkk]-d[kkk]);
}
}

}