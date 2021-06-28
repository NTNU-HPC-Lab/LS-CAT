//
// Created by mulholbn on 5/1/20.
//

#ifndef CUDAIMP_TYPES_H
#define CUDAIMP_TYPES_H

#include <stdint.h>


//WIDTH AND HEIGHT MUST BE THE SAME
#define WIDTH 4
#define HEIGHT 4
//Set SIZE to equal WIDTH * HEIGHT. Saves processing power.
#define SIZE 16
//Number of iterations to search through
#define NUMITERATIONS 8
//DO NOT CHANGE
#define NUMMOVES 8




typedef enum move_t {
    up = 0,
    down = 1,
    left = 2,
    right = 3
}Move;


typedef uint16_t Board[HEIGHT][WIDTH];
typedef Move MoveList[NUMMOVES];

typedef enum status_t {
    boardUpdated = 0,
    boardUnchanged = 1,
    boardFull = 2
}status;

typedef struct bStatus{
    Board b;
    status s;
}boardStatus;





#endif //CUDAIMP_TYPES_H
