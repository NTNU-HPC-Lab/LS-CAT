#include "includes.h"
//	INCLUDES

// CUDA

// GIS

/**
* 	PARS
*/
#define 					BLOCK_DIM_small				64
#define 					BLOCK_DIM 					256

static const unsigned int 	threads 					= 512;
bool 						print_intermediate_arrays 	= false;
const char 					*BASE_PATH 					= "/home/giuliano/git/cuda/reduction";

/*
*	kernel labels
*/
const char		*kern_0			= "filter_roi";
const char 		*kern_1 		= "imperviousness_change_histc_sh_4"	;
const char 		*kern_2 		= "imperviousness_change"	;
char			buffer[255];

/*
* 		DEFINE I/O files
*/
// I/–
//const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/reduction/data/ROI.tif";
//const char 		*FIL_BIN1 		= "/home/giuliano/git/cuda/reduction/data/BIN1.tif";
//const char 		*FIL_BIN2 		= "/home/giuliano/git/cuda/reduction/data/BIN2.tif";
const char 		*FIL_ROI 		= "/media/DATI/db-backup/ssgci-data/testing/ssgci_roi.tif";
const char 		*FIL_BIN1 		= "/media/DATI/db-backup/ssgci-data/testing/ssgci_bin.tif";
const char 		*FIL_BIN2 		= "/media/DATI/db-backup/ssgci-data/testing/ssgci_bin2.tif";

// –/O
const char 		*FIL_LTAKE_grid	= "/home/giuliano/git/cuda/reduction/data/LTAKE_map.tif";
const char 		*FIL_LTAKE_count= "/home/giuliano/git/cuda/reduction/data/LTAKE_count.txt";

/*	+++++DEFINEs+++++	*/
__global__ void filter_roi( unsigned char *BIN, const unsigned char *ROI, unsigned int map_len){
unsigned int tid 		= threadIdx.x;
unsigned int bix 		= blockIdx.x;
unsigned int bdx 		= blockDim.x;
unsigned int gdx 		= gridDim.x;
unsigned int i 			= bix*bdx + tid;
unsigned int gridSize 	= bdx*gdx;

while (i < map_len)
{
//BIN[i] *= ROI[i];
BIN[i] = (unsigned char) ((int)BIN[i] * (int)ROI[i]);
i += gridSize;
}
}