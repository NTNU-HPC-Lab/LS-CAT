#include "includes.h"
#define L2HYS_EPSILON 		0.01f
#define L2HYS_EPSILONHYS	1.0f
#define L2HYS_CLIP			0.2f
#define data_h2y            30
//long h_windowx=Imagewidth/Windowx;
//long h_windowy=ImageHeight/Windowy;
//dim3 blocks(h_windowx,h_windowy);//h_windowx=ImageWidth/Windowx,h_windowy=ImageHeight/Windowy
//dim3 threads(Windowx,Windowy);//Ã¿Ò»¸öÏß³Ì¿é¼ÆËãÒ»¸öcellµÄÌØÕ÷Á¿

//dim3 block(18,7);//Ò»¸öcell·Ö18¸ö½Ç¶È·½Ïò,Ò»¸ö·½Ïò7¸öcell£¬
__global__ void smoothcell(float *in,float *out){
int t_nleft,t_nright;
t_nleft=(threadIdx.x-1+10)%10;
t_nright=(threadIdx.x+1)%10;
float *t_ptemp,t_ftemp[10];
t_ptemp=in+blockIdx.x*70+blockIdx.y*10;//+threadIdx.y)*0.8f+0.1f*(in+blockIdx.x*70+threadIdx.x*10+t_left)
/*__syncthreads();*/
if(t_ptemp)
t_ftemp[threadIdx.x]=t_ptemp[threadIdx.x]*0.8f+0.1f*t_ptemp[t_nleft]+0.1f*t_ptemp[t_nright];
__syncthreads();
out[blockIdx.x*70+blockIdx.y*10+threadIdx.x]=t_ftemp[threadIdx.x];
__syncthreads();
}