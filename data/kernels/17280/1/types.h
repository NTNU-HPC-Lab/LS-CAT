#include <stdint.h>


//WIDTH AND HEIGHT MUST BE THE SAME
#define WIDTH 4
#define HEIGHT 4
//Set SIZE to equal WIDTH * HEIGHT. Saves processing power.
#define SIZE 16
//Number of iterations to search through
#define NUMITERATIONS 8
//Number of moves
#define NUMMOVES 4




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



