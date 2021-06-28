/*
 * Constants.h
 *
 *  Created on: Sep 19, 2018
 *      Author: dpalominop
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_


#define O_TILE_HEIGHT 16
#define O_TILE_WIDTH 16
#define KERNEL_LENGTH 5

#define BLOCK_DIM_X O_TILE_WIDTH+(KERNEL_LENGTH/2)*2
#define BLOCK_DIM_Y O_TILE_HEIGHT+(KERNEL_LENGTH/2)*2


#endif /* CONSTANTS_H_ */
