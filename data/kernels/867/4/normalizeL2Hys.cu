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
__global__ void normalizeL2Hys(float *in,float *out)
{
int bid=blockIdx.x;
int tid=threadIdx.x;
// Sum the vector
float sum = 0;

float *t_ftemp=in+bid*30;
float *t_foutemp=out+bid*30;
sum+=t_ftemp[tid]*t_ftemp[tid];
__syncthreads();
// Compute the normalization term
float norm = 1.0f/(rsqrt(sum) + L2HYS_EPSILONHYS * 30);
t_foutemp[tid]=t_ftemp[tid]*norm;
__syncthreads();


}