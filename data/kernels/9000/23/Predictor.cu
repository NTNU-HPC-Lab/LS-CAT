#include "includes.h"
__global__ void Predictor (const double TIME, double4 *p_pred, float4  *v_pred, float4  *a_pred, double4 *p_corr, double4 *v_corr, double  *loc_time, double4 *acc, double4 *acc1, double4 *acc2, double4 *acc3, int istart, int* nvec, int ppgpus, unsigned int N){



int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
int cost = ppgpus+istart;

if(i>=cost){
i = nvec[i - cost];
if(i>=istart && i < cost)
i=-1;
}
if(i<0)
return;

double timestep = TIME - loc_time[i];
double t2 = timestep * timestep;
double t3 = t2 * timestep;
double t4 = t2 * t2;
double t5 = t4 * timestep;

t2 *= 0.5;
t3 *= 0.1666666666666666666666;
t4 *= 0.0416666666666666666666;
t5 *= 0.0083333333333333333333;

double4 myppred;
myppred.x = p_pred[i].x;
myppred.y = p_pred[i].y;
myppred.z = p_pred[i].z;

float4  mypred;
mypred.x = v_pred[i].x;
mypred.y = v_pred[i].y;
mypred.z = v_pred[i].z;

double4 mypcorr;
mypcorr.x = p_corr[i].x;
mypcorr.y = p_corr[i].y;
mypcorr.z = p_corr[i].z;

double4 myvcorr;
myvcorr.x = v_corr[i].x;
myvcorr.y = v_corr[i].y;
myvcorr.z = v_corr[i].z;

double4 myacc;
myacc.x = acc[i].x;
myacc.y = acc[i].y;
myacc.z = acc[i].z;

double4 myacc1;
myacc1.x = acc1[i].x;
myacc1.y = acc1[i].y;
myacc1.z = acc1[i].z;

double4 myacc2;
myacc2.x = acc2[i].x;
myacc2.y = acc2[i].y;
myacc2.z = acc2[i].z;

double4 myacc3;
myacc3.x = acc3[i].x;
myacc3.y = acc3[i].y;
myacc3.z = acc3[i].z;


myppred.x = mypcorr.x + timestep * myvcorr.x +
t2 * myacc.x  +
t3 * myacc1.x +
t4 * myacc2.x +
t5 * myacc3.x ;

myppred.y = mypcorr.y + timestep * myvcorr.y +
t2 * myacc.y  +
t3 * myacc1.y +
t4 * myacc2.y +
t5 * myacc3.y ;

myppred.z = mypcorr.z + timestep * myvcorr.z +
t2 * myacc.z  +
t3 * myacc1.z +
t4 * myacc2.z +
t5 * myacc3.z ;

p_pred[i].x = myppred.x;
p_pred[i].y = myppred.y;
p_pred[i].z = myppred.z;

mypred.x = myvcorr.x + timestep * myacc.x +
t2 * myacc1.x +
t3 * myacc2.x +
t4 * myacc3.x ;

mypred.y = myvcorr.y + timestep * myacc.y +
t2 * myacc1.y +
t3 * myacc2.y +
t4 * myacc3.y ;

mypred.z = myvcorr.z + timestep * myacc.z +
t2 * myacc1.z +
t3 * myacc2.z +
t4 * myacc3.z ;

v_pred[i].x = mypred.x;
v_pred[i].y = mypred.y;
v_pred[i].z = mypred.z;

mypred.x = myacc.x + timestep * myacc1.x +
t2 * myacc2.x +
t3 * myacc3.x ;

mypred.y = myacc.y + timestep * myacc1.y +
t2 * myacc2.y +
t3 * myacc3.y ;

mypred.z = myacc.z + timestep * myacc1.z +
t2 * myacc2.z +
t3 * myacc3.z ;

a_pred[i].x = mypred.x;
a_pred[i].y = mypred.y;
a_pred[i].z = mypred.z;
}