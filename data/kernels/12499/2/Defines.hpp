#pragma once

typedef float p_type;	//type to use for all particle stuff

#define USE_CPU

#define BIG_G .00006674

#define EARTH_KG 100

#define MAX_FORCE BIG_G * EARTH_KG /5

//#define SPIRAL_DIST

//#define DISK_DIST

//#define GALAXY_DIST

#define PI 3.141592

#define E_MATH 2.71828

#define TPB 1024	//threads to use per block on gpu

//1million km per pixel
//billion dist units per pixel
//earth density 5513 kg/m3
//thousand trillion kg per 1u mass