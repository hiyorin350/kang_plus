#define MAX_FILENAME   256  /* 想定するファイル名の最大長 */
#define MAX_BUFFERSIZE 256  /* 利用するバッファ最大長 */

typedef unsigned char  BYTE;   /* 1byte符号なし整数 */
typedef unsigned short WORD;   /* 2byte符号なし整数 */
typedef unsigned long  DWORD;  /* 4byte符号なし整数 */
typedef long           LONG;   /* 4byte整数 */

typedef struct tagPGMStruct{
  WORD t;  /* type */
  LONG w;  /* width */
  LONG h;  /* height */
  LONG m;  /* maximum gradation value */
  BYTE **p;  /* pixel value */
}PGMStruct;

typedef struct tagPPMStruct{
  WORD t;  /* type */
  LONG w;  /* width */
  LONG h;  /* height */
  LONG m;  /* maximum gradation value */
  BYTE ***p;  /* pixel value */
}PPMStruct;

PGMStruct* PGMCreate(long width, long height, long max);
void PGMFree(PGMStruct *img);
PGMStruct* PGMRead(char *fname);
int PGMWrite(PGMStruct *img, char *fname);

PPMStruct* PPMCreate(long width, long height, long max);
void PPMFree(PPMStruct *img);
PPMStruct* PPMRead(char *fname);
int PPMWrite(PPMStruct *img, char *fname);