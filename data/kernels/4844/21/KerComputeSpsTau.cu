#include "includes.h"
__global__ void KerComputeSpsTau(unsigned n,unsigned pini,float smag,float blin ,const float4 *velrhop,const float2 *gradvelff,float2 *tauff)
{
unsigned p=blockIdx.x*blockDim.x + threadIdx.x;
if(p<n){
const unsigned p1=p+pini;
float2 rr=gradvelff[p1*3];   const float grad_xx=rr.x,grad_xy=rr.y;
rr=gradvelff[p1*3+1]; const float grad_xz=rr.x,grad_yy=rr.y;
rr=gradvelff[p1*3+2]; const float grad_yz=rr.x,grad_zz=rr.y;
const float pow1=grad_xx*grad_xx + grad_yy*grad_yy + grad_zz*grad_zz;
const float prr= grad_xy*grad_xy + grad_xz*grad_xz + grad_yz*grad_yz + pow1+pow1;
const float visc_sps=smag*sqrt(prr);
const float div_u=grad_xx+grad_yy+grad_zz;
const float sps_k=(2.0f/3.0f)*visc_sps*div_u;
const float sps_blin=blin*prr;
const float sumsps=-(sps_k+sps_blin);
const float twovisc_sps=(visc_sps+visc_sps);
float one_rho2=1.0f/velrhop[p1].w;
//-Computes new values of tau[].
const float tau_xx=one_rho2*(twovisc_sps*grad_xx +sumsps);
const float tau_xy=one_rho2*(visc_sps   *grad_xy);
tauff[p1*3]=make_float2(tau_xx,tau_xy);
const float tau_xz=one_rho2*(visc_sps   *grad_xz);
const float tau_yy=one_rho2*(twovisc_sps*grad_yy +sumsps);
tauff[p1*3+1]=make_float2(tau_xz,tau_yy);
const float tau_yz=one_rho2*(visc_sps   *grad_yz);
const float tau_zz=one_rho2*(twovisc_sps*grad_zz +sumsps);
tauff[p1*3+2]=make_float2(tau_yz,tau_zz);
}
}