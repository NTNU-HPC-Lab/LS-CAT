#pragma once

typedef double p_type;	//type to use for all particle stuff

//#define SAVE_IMAGES
//#define SAVE_DATA
//#define LOAD_FILE



#define COORD_TO_PIXEL 100

#define EARTH_KG 400000000

#define PI 3.141592

#define E_MATH 2.71828

#define TPB 1024	//threads to use per block on gpu
#define TPB2d 16

#define USE_MOUSE
#define MOUSE_MASS	EARTH_KG*100000
#define MOUSE_SCALE 150
static int* mousePos;

#define MAX_VEL 4400

//1million km per pixel
//billion dist units per pixel
//earth density 5513 kg/m3
//thousand trillion kg per 1u mass