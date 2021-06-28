#include "includes.h"
__global__ void Corrector_gpu(double GTIME, double *local_time, double *step, int *next, unsigned long nextsize, double4 *pos_CH, double4 *vel_CH, double4 *a_tot_D, double4 *a1_tot_D, double4 *a2_tot_D, double4 *a_H0, double4 *a3_H, double ETA6, double ETA4, double DTMAX, double DTMIN, unsigned int N){

unsigned int gtid = blockIdx.x*blockDim.x + threadIdx.x;

double dt;
int who = next[gtid];
int who1 = gtid + nextsize;
int who2 = who1 + nextsize;

if(gtid >= nextsize )
return;

a_H0[gtid].w = a_H0[gtid].x * a_H0[gtid].x +
a_H0[gtid].y * a_H0[gtid].y +
a_H0[gtid].z * a_H0[gtid].z ;

a_H0[who1].w = a_H0[who1].x * a_H0[who1].x +
a_H0[who1].y * a_H0[who1].y +
a_H0[who1].z * a_H0[who1].z ;

a_H0[who2].w =  a_H0[who2].x * a_H0[who2].x +
a_H0[who2].y * a_H0[who2].y +
a_H0[who2].z * a_H0[who2].z ;

double h = GTIME-local_time[who];
local_time[who] = GTIME;

double h1 = 0.5*h;
double h2 = h1*h1;
double h3 = 0.75/(h1*h1*h1);
double h4 = 1.5/(h2*h2);
double h5 = 7.5/(h2*h2*h1);

double Amin = a_H0[gtid].x - a_tot_D[who].x;
double Aplu = a_H0[gtid].x + a_tot_D[who].x;
double Jmin = h1 * (a_H0[who1].x - a1_tot_D[who].x);
double Jplu = h1 * (a_H0[who1].x + a1_tot_D[who].x);
double Smin = h1 * h1 * (a_H0[who2].x - a2_tot_D[who].x);
double Splu = h1 * h1 * (a_H0[who2].x + a2_tot_D[who].x);

double over= 1.0/15.0;

pos_CH[who].x = pos_CH[who].x + h1*vel_CH[who].x - 0.4*h2*Amin + over*h2*Jplu;
vel_CH[who].x = vel_CH[who].x + h1*Aplu          - 0.4*h1*Jmin + over*h1*Splu;
pos_CH[who].x += h1*vel_CH[who].x;

a3_H[who].x = h3*(-5.0*Amin + 5.0*Jplu - Smin);
double a4halfx = h4*(-Jmin + Splu);
double a5halfx = h5*(3.0*Amin - 3.0*Jplu + Smin);
a3_H[who].x += h1*a4halfx + 0.5*h2*a5halfx;
a4halfx += h1*a5halfx;

Amin = a_H0[gtid].y - a_tot_D[who].y;
Aplu = a_H0[gtid].y + a_tot_D[who].y;
Jmin = h1 * (a_H0[who1].y - a1_tot_D[who].y);
Jplu = h1 * (a_H0[who1].y + a1_tot_D[who].y);
Smin = h1 * h1 * (a_H0[who2].y - a2_tot_D[who].y);
Splu = h1 * h1 * (a_H0[who2].y + a2_tot_D[who].y);

pos_CH[who].y = pos_CH[who].y + h1*vel_CH[who].y - 0.4*h2*Amin + over*h2*Jplu;
vel_CH[who].y = vel_CH[who].y + h1*Aplu          - 0.4*h1*Jmin + over*h1*Splu;
pos_CH[who].y += h1*vel_CH[who].y;

a3_H[who].y = h3*(-5.0*Amin + 5.0*Jplu - Smin);
double a4halfy = h4*(-Jmin + Splu);
double a5halfy = h5*(3.0*Amin - 3.0*Jplu + Smin);
a3_H[who].y += h1*a4halfy + 0.5*h2*a5halfy;
a4halfy += h1*a5halfy;

Amin = a_H0[gtid].z - a_tot_D[who].z;
Aplu = a_H0[gtid].z + a_tot_D[who].z;
Jmin = h1 * (a_H0[who1].z - a1_tot_D[who].z);
Jplu = h1 * (a_H0[who1].z + a1_tot_D[who].z);
Smin = h1 * h1 * (a_H0[who2].z - a2_tot_D[who].z);
Splu = h1 * h1 * (a_H0[who2].z + a2_tot_D[who].z);

pos_CH[who].z = pos_CH[who].z + h1*vel_CH[who].z - 0.4*h2*Amin + over*h2*Jplu;
vel_CH[who].z = vel_CH[who].z + h1*Aplu          - 0.4*h1*Jmin + over*h1*Splu;
pos_CH[who].z += h1*vel_CH[who].z;

a3_H[who].z = h3*(-5.0*Amin + 5.0*Jplu - Smin);
double a4halfz = h4*(-Jmin + Splu);
double a5halfz = h5*(3.0*Amin - 3.0*Jplu + Smin);
a3_H[who].z += h1*a4halfz + 0.5*h2*a5halfz;
a4halfz += h1*a5halfz;

a3_H[who].w = sqrt(a3_H[who].x*a3_H[who].x + a3_H[who].y*a3_H[who].y + a3_H[who].z*a3_H[who].z);
double a4mod = sqrt(a4halfx*a4halfx + a4halfy*a4halfy + a4halfz*a4halfz);
double a5mod = sqrt(a5halfx*a5halfx + a5halfy*a5halfy + a5halfz*a5halfz);

double    dt6 = (sqrt(a_H0[gtid].w*a_H0[who2].w) + a_H0[who1].w) / (a5mod*a3_H[who].w + a4mod*a4mod);
dt6 = ETA6 * pow(dt6,1.0/6.0);

double stp = h;
double overh3 = 1.0/(stp*stp*stp);
double overh2 = 1.0/(stp*stp);

double a2dx = overh2 * (-6.0 * (a_tot_D[who].x - a_H0[gtid].x) -
stp * (4.0 * a_H0[who1].x + 2.0 * a1_tot_D[who].x));
double a2dy = overh2 * (-6.0 * (a_tot_D[who].y - a_H0[gtid].y) -
stp * (4.0 * a_H0[who1].y + 2.0 * a1_tot_D[who].y));
double a2dz = overh2 * (-6.0 * (a_tot_D[who].z - a_H0[gtid].z) -
stp * (4.0 * a_H0[who1].z + 2.0 * a1_tot_D[who].z));

double a3dx = overh3 * (12.0 * (a_tot_D[who].x - a_H0[gtid].x) +
6.0 * stp * (a_H0[who1].x + a1_tot_D[who].x));
double a3dy = overh3 * (12.0 * (a_tot_D[who].y - a_H0[gtid].y) +
6.0 * stp * (a_H0[who1].y + a1_tot_D[who].y));
double a3dz = overh3 * (12.0 * (a_tot_D[who].z - a_H0[gtid].z) +
6.0 * stp * (a_H0[who1].z + a1_tot_D[who].z));

a2dx += h*a3dx;
a2dy += h*a3dy;
a2dx += h*a3dz;

a_H0[who2].w =  a2dx*a2dx + a2dy*a2dy + a2dz*a2dz;
a3_H[who].w = a3dx*a3dx + a3dy*a3dy + a3dz*a3dz;

double dt4 = sqrt(ETA4*(sqrt(a_H0[gtid].w*a_H0[who2].w) + a_H0[who1].w) / (sqrt(a_H0[who1].w*a3_H[who].w) + a_H0[who2].w));

dt = 0.5*dt4+0.5*dt6;

double rest = GTIME / (2.0 * step[who]);
rest = (double)((int)(rest)) - rest;

//	return;
//	pos_CH[who].x = step[who];
//	return;

if(dt > 2.0*step[who] && rest == 0.0 && 2.0*step[who] <= DTMAX)
step[who] *= 2.0;
else if (dt < 0.5*step[who])
step[who] *= 0.25;
else if (dt < step[who])
step[who]*=0.5;

if(step[who] < DTMIN)
step[who] = DTMIN;

a_tot_D[who] = a_H0[gtid];
a1_tot_D[who] = a_H0[who1];
a2_tot_D[who] = a_H0[who2];

}