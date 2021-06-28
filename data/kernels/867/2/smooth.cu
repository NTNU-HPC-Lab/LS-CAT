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
__global__ void smooth(float *in,float *out)
{
int k,j,i;
int m_nBIN=10;
float *m_pCellFeatures=in;
int t_nLineWidth=70;
float t_pTemp[10];
for ( k = 0; k < 18; ++k )//18
{
for ( j = 0; j < 7; ++j )//7
{
for ( i = 0; i< 10; ++i )//10
{
int t_nLeft;
int t_nRight;
t_nLeft = ( i - 1 + m_nBIN ) % m_nBIN;
t_nRight = ( i + 1 ) % m_nBIN;

t_pTemp[i] = m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + i] * 0.8f
+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + t_nLeft] * 0.1f
+ m_pCellFeatures[k * t_nLineWidth + j * m_nBIN + t_nRight] * 0.1f;
}

for ( i = 0; i < m_nBIN; ++i )
{
out[k * t_nLineWidth + j * m_nBIN + i] = t_pTemp[i];
}
}
}

}