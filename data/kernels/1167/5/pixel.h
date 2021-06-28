#ifndef __PIXEL_H_
#define __PIXEL_H_
#include <cstdint>

typedef struct Pixel_t{
  uint8_t r,g,b;

} Pixel;
/*class Pixel
{
  public:
    uint8_t r,g,b;
    Pixel() = default;
    Pixel(uint8_t x,uint8_t y, uint8_t z){
      r=x;
      g=y;
      b=z;
    }
};
*/
#endif
