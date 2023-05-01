/*
 https://codegolf.stackexchange.com/questions/35569/tweetable-mathematical-art
 g++ tweetart.cpp -o tweetart
 Original version: http://pastebin.com/uQkCQGhz
 Cleaned up version: http://pastebin.com/index/uQkCQGhz
*/
#include <stdio.h>
#include <math.h>
#include <stdint.h> // uint8_t
#define DIM 1024
#define DM1 (DIM-1)
#define MAX_COLORS 255 // or 255, or 65535
#define _sq(x) ((x)*(x))                           // square
#define _cb(x) abs((x)*(x)*(x))                    // absolute value of cube
#define _cr(x) (unsigned short)(pow((x),1.0/3.0))  // cube root

typedef uint8_t  u8;
typedef uint16_t u16;
u16 GR(int,int);
u16 BL(int,int);

#define r(n)(rand()%n)

u16 RD(int i,int j){
    static char c[1024][1024];return!c[i][j]?c[i][j]=!r(999)?r(256):RD((i+r(2))%1024,(j+r(2))%1024):c[i][j];
}
u16 GR(int i,int j){
    static char c[1024][1024];return!c[i][j]?c[i][j]=!r(999)?r(256):GR((i+r(2))%1024,(j+r(2))%1024):c[i][j];
}
u16 BL(int i,int j){
    static char c[1024][1024];return!c[i][j]?c[i][j]=!r(999)?r(256):BL((i+r(2))%1024,(j+r(2))%1024):c[i][j];
}

FILE *fp;
void pixel_write(int i, int j){
    static u16 color[3];
    color[0] = RD(i,j) & MAX_COLORS;
    color[1] = GR(i,j) & MAX_COLORS;
    color[2] = BL(i,j) & MAX_COLORS;
    if (MAX_COLORS < 256){
        color[0] &= 0xFF;
        color[0] |= ((color[1] & 0xFF) << 8);
        color[1] = color[2];
    }
    // 8-bit = 3 bytes
    //16-bit = 6 bytes
    fwrite(color, 1+(MAX_COLORS>255), 3, fp);
}

int main(){
    fp = fopen("mathpic.ppm","wb");
    if( !fp ) return -1;
    fprintf(fp, "P6\n%d %d\n%d\n", DIM, DIM, MAX_COLORS);
    for(int j=0;j<DIM;j++)
        for(int i=0;i<DIM;i++)
            pixel_write(i,j);
    fclose(fp);
    return 0;
}

