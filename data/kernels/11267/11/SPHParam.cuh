#ifndef SPHPARAM_CUH
#define SPHPARAM_CUH
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

using namespace glm;

class SPHParam{
public:
	float restDensity;// (){ return 1000; }
	float gasConstant;// (){ return 1; }
	float viscosityCoeff;// (){ return 6.5f; }
	float timeStep;// (){ return 10 / 80; }
	float surfaceTreshold;// (){ return 6.0f; }
	float surfaceTensionCoeff;// (){ return 0.0728; }
	vec3 gravity;// (){ return vec3(0.0f, 9.8f, 0.0f); }
	float mParticleRadius;// (){ return 0.5; }
	float wallDamping;// (){ return 0.2; }
	int particleSizeReserver;// (){ return 100; }
	float kernelSize;
	int maxParticleCount;
	int particleCounter;
	float HIrandomizer;
	float LOrandomizer;
	int cudaThreadsPerblock;
	vec4 surfaceColor;
	vec4 insideColor;
};

#endif