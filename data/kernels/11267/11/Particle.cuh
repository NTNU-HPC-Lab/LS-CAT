#ifndef PARTICLE_CUH
#define PARTICLE_CUH

#include <math.h>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
//#include <glm/detail/type_vec4.hpp>

using namespace glm;

class Particle
{
public:
	//Particle(){};
	vec3	mPosition;   ///< Position (in world units) of center of particle
	vec3	mVelocity;   ///< Velocity of particle	
	vec4	color;
	//vec3	mAcc;		///< Accelaration of particles
	float	mMass;   ///< Mass of particle
	//float	mSize;   ///< Size of particle
	//int     mBirthTime;   ///< Birth time of particle, in "ticks
	float	mDensity;
	float	mPressure;
	vec3	mForces;
	vec3	colorFieldNormal;
	float	colorFieldLaplacian;
	bool	ghost;
	//bool	mSurfaceFlag;
};

#endif