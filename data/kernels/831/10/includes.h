const int TILE_WIDTH	= 16;
const int TILE_HEIGHT	= 16;
const int FILTER_RADIUS = 3; //  3 for averge, 1 for sobel
const int FILTER_AREA	= (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1);
const int BLOCK_WIDTH	= TILE_WIDTH + 2 * FILTER_RADIUS;
const int BLOCK_HEIGHT	= TILE_HEIGHT + 2 * FILTER_RADIUS;
//new series 
