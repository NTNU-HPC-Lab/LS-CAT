#include "includes.h"
__global__ void KerFtCalcForcesRes(unsigned ftcount,bool simulate2d,double dt ,const float3 *ftoomega,const float3 *ftovel,const double3 *ftocenter,const float3 *ftoforces ,float3 *ftoforcesres,double3 *ftocenterres)
{
const unsigned cf=blockIdx.x*blockDim.x + threadIdx.x; //-Floating number.
if(cf<ftcount){
//-Compute fomega.
float3 fomega=ftoomega[cf];
{
const float3 omegaace=ftoforces[cf*2+1];
fomega.x=float(dt*omegaace.x+fomega.x);
fomega.y=float(dt*omegaace.y+fomega.y);
fomega.z=float(dt*omegaace.z+fomega.z);
}
float3 fvel=ftovel[cf];
//-Zero components for 2-D simulation. | Anula componentes para 2D.
float3 face=ftoforces[cf*2];
if(simulate2d){ face.y=0; fomega.x=0; fomega.z=0; fvel.y=0; }
//-Compute fcenter.
double3 fcenter=ftocenter[cf];
fcenter.x+=dt*fvel.x;
fcenter.y+=dt*fvel.y;
fcenter.z+=dt*fvel.z;
//-Compute fvel.
fvel.x=float(dt*face.x+fvel.x);
fvel.y=float(dt*face.y+fvel.y);
fvel.z=float(dt*face.z+fvel.z);
//-Store data to update floating. | Guarda datos para actualizar floatings.
ftoforcesres[cf*2]=fomega;
ftoforcesres[cf*2+1]=fvel;
ftocenterres[cf]=fcenter;
}
}