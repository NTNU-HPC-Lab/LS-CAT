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
__global__ void countblock(float *in ,float *out)
{
//if(in+70*blockIdx.x+(blockIdx.y+threadIdx.x)*10!=NULL)
//{
float *ptr_in=in+70*blockIdx.x+(blockIdx.y+threadIdx.x)*10;//threadIdx.x;//70=Ò»¸ö½Ç¶È·½Ïò7¸öcell£¬Ã¿¸öcell 10¸öbin,
float *ptr_out=out+120*blockIdx.x+30*blockIdx.y+10*threadIdx.x;//threadIdx.x;//Ò»¸ö½Ç¶È·½Ïò4¸öblock£¬Ò»¸öblock3¸öcell£¬Ò»¸öcell 10¸öbin,
//Ò»¸öblock3¸öcell£¬Ò»¸öcell 10¸öbin,
ptr_out[threadIdx.y]=ptr_in[threadIdx.y];
////}
}