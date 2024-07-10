#include <stdio.h>
#include <stdlib.h>
#include "pnmio.h"


PGMStruct* PGMCreate(long width, long height, long max)
{
  int i;
  PGMStruct *img;

  if(width<=0 || height<=0){
//	puts("# width>0，height>0としてください");
	exit(EXIT_FAILURE);
  }
  if(max<=0 || max>255){
//	puts("# 0<max<=255としてください");
	exit(EXIT_FAILURE);
  }

  img=(PGMStruct *)malloc(sizeof(PGMStruct));
  if(img==NULL){
//	puts("# メモリ確保失敗");
	exit(EXIT_FAILURE);
  }
  //sprintf(img->type,"P5");
  //img->type[0]='P';
  //img->type=type;
  img->t=5;
  img->w=width;
  img->h=height;
  img->m=max;

  /* 画像データ領域の確保 */
  img->p=(BYTE **)malloc(sizeof(BYTE *)*height);
  if(img->p==NULL){
//	puts("# メモリ領域確保失敗");
	exit(EXIT_FAILURE);
  }
  for(i=0;i<height;i++){
	img->p[i]=(BYTE *)malloc(sizeof(BYTE)*width);
	if(img->p[i]==NULL){
//	  puts("# メモリ領域確保失敗");
	  exit(EXIT_FAILURE);
	}
  }

  return(img);
}


void PGMFree(PGMStruct *img)
{
  int i;

  for(i=0;i<img->h;i++) free(img->p[i]);
  free(img->p);

  free(img);
}


PGMStruct* PGMRead(char *fname)
{
  char buffer[MAX_BUFFERSIZE];   /*データ読み込み用作業変数 */
  int i,x,y;
  FILE *fp;                      /*ファイルポインタ*/
  PGMStruct *img;

  if((fp=fopen(fname,"rb")) == NULL ){  /* 入力ファイルのオープン */
//	puts( "# その名前のファイルは存在しません" );
	exit(EXIT_FAILURE);
  }

  /*ファイルタイプ(=P5)の確認*/
  if(fgets(buffer,MAX_BUFFERSIZE,fp)==NULL){
//	puts("PGMファイルを開くことができません");  //必要か？
  }else{
	if(buffer[0]!='P' || buffer[1]!='5'){
//	  puts("# ファイルのフォーマットがP5とは異なります");
	  exit(EXIT_FAILURE);
	}
  }

  img=(PGMStruct *)malloc(sizeof(PGMStruct));
  if(img==NULL){
//	puts("# メモリ確保失敗");
	exit(EXIT_FAILURE);
  }
  img->t=5;

  /* 画像の横幅・縦幅の読込(#から始まるコメントは読み飛ばす)*/
  img->w=0;
  img->h=0;
  while(img->w==0 || img->h==0){
	fgets(buffer,MAX_BUFFERSIZE,fp);
	if(buffer[0]!='#'){
	  sscanf(buffer,"%ld %ld",&img->w,&img->h);
	}
  }
  /* 画像の最大階調値の読込(#から始まるコメントは読み飛ばす)*/
  img->m=0;
  while(img->m==0){
	fgets(buffer,MAX_BUFFERSIZE,fp);
	if(buffer[0]!='#')
	  sscanf(buffer,"%ld",&img->m);
  }

  /* 画像データ領域の確保 */
  img->p=(BYTE **)malloc(sizeof(BYTE *)*img->h);
  if(img->p==NULL){
//	puts("# メモリ領域確保失敗");
	exit(EXIT_FAILURE);
  }
  for(i=0;i<img->h;i++){
	img->p[i]=(BYTE *)malloc(sizeof(BYTE)*img->w);
	if(img->p[i]==NULL){
//	  puts("# メモリ領域確保失敗");
	  exit(EXIT_FAILURE);
	}
  }

  /*画像データを読み込んで画像用配列に代入する*/
  for(y=0;y<img->h;y++){
	for(x=0;x<img->w;x++){
	  img->p[y][x]=(BYTE)fgetc(fp);
	}
  }
  fclose(fp);

  return(img);
}


int PGMWrite(PGMStruct *img, char *fname)
{
  FILE *fp;  /* ファイルポインタ */
  int x,y;   /* ループ変数 */

  switch(img->t){
  case 5:
	if((fp=fopen(fname,"wb"))==NULL){  /* 出力ファイルのオープン */
//	  puts("# ファイルオープン失敗");
	  exit(EXIT_FAILURE);
	}
	fputs("P5\n",fp); /* ファイル識別子"P5"を先頭に出力する */
	fputs("# Created by pnmio.lib\n",fp);  /* #で始まるコメント行（省略可能） */
	fprintf(fp,"%ld %ld\n",img->w,img->h);  /* 画像の横幅・縦幅の出力 */
	fprintf(fp,"%ld\n",img->m);  /* 最大階調値の出力 */
	for(y=0;y<img->h;y++){  /* 画像データの出力 */
	  for(x=0;x<img->w;x++){
		fputc(img->p[y][x],fp);
	  }
	}
	fclose(fp);
	break;

  default:
//	puts("# ファイルタイプの設定が不適切です");
	return(1);
  }

  return(0);
}


PPMStruct* PPMCreate(long width, long height, long max)
{
  int i,k;
  PPMStruct *img;

  if(width<=0 || height<=0){
//	puts("# width>0，height>0としてください");
	exit(EXIT_FAILURE);
  }
  if(max<=0 || max>255){
//	puts("# 0<max<=255としてください");
	exit(EXIT_FAILURE);
  }

  img=(PPMStruct *)malloc(sizeof(PPMStruct));
  if(img==NULL){
//	puts("# メモリ確保失敗");
	exit(EXIT_FAILURE);
  }
  //sprintf(img->type,"P6");
  //img->type[0]='P';
  //img->type=type;
  img->t=6;
  img->w=width;
  img->h=height;
  img->m=max;

  /* 画像データ領域の確保 */
  img->p=(BYTE ***)malloc(sizeof(BYTE **)*3*height);
  if(img->p==NULL){
//	puts("# メモリ領域確保失敗");
	exit(EXIT_FAILURE);
  }
  for(k=0;k<3;k++){
	img->p[k]=(BYTE **)malloc(sizeof(BYTE)*height*width);
	if(img->p[k]==NULL){
//	  puts("# メモリ領域確保失敗");
	  exit(EXIT_FAILURE);
	}
	for(i=0;i<height;i++){
	  img->p[k][i]=(BYTE *)malloc(sizeof(BYTE)*width);
	  if(img->p[k][i]==NULL){
//		puts("# メモリ領域確保失敗");
		exit(EXIT_FAILURE);
	  }
	}
  }

  return(img);
}


void PPMFree(PPMStruct *img)
{
  int i,c;

  for(c=0;c<3;c++){
	for(i=0;i<img->h;i++) free(img->p[c][i]);
	free(img->p[c]);
  }
  free(img->p);

  free(img);
}


PPMStruct* PPMRead(char *fname)
{
  char buffer[MAX_BUFFERSIZE];   /*データ読み込み用作業変数 */
  int i,k,c,x,y;
  FILE *fp;                      /*ファイルポインタ*/
  PPMStruct *img;

  if((fp=fopen(fname,"rb")) == NULL ){  /* 入力ファイルのオープン */
//	puts( "# その名前のファイルは存在しません" );
	exit(EXIT_FAILURE);
  }

  /*ファイルタイプ(=P6)の確認*/
  if(fgets(buffer,MAX_BUFFERSIZE,fp)==NULL){
//	puts("PPMファイルを開くことができません");  //必要か？
  }else{
	if(buffer[0]!='P' || buffer[1]!='6'){
//	  puts("# ファイルのフォーマットがP6とは異なります");
	  exit(EXIT_FAILURE);
	}
  }

  img=(PPMStruct *)malloc(sizeof(PPMStruct));
  if(img==NULL){
//	puts("# メモリ確保失敗");
	exit(EXIT_FAILURE);
  }
  img->t=6;

  /* 画像の横幅・縦幅の読込(#から始まるコメントは読み飛ばす)*/
  img->w=0;
  img->h=0;
  while(img->w==0 || img->h==0){
	fgets(buffer,MAX_BUFFERSIZE,fp);
	if(buffer[0]!='#'){
	  sscanf(buffer,"%ld %ld",&img->w,&img->h);
	}
  }
  /* 画像の最大階調値の読込(#から始まるコメントは読み飛ばす)*/
  img->m=0;
  while(img->m==0){
	fgets(buffer,MAX_BUFFERSIZE,fp);
	if(buffer[0]!='#')
	  sscanf(buffer,"%ld",&img->m);
  }

  /* 画像データ領域の確保 */
  img->p=(BYTE ***)malloc(sizeof(BYTE **)*3*img->h);
  if(img->p==NULL){
//	puts("# メモリ領域確保失敗");
	exit(EXIT_FAILURE);
  }
  for(k=0;k<3;k++){
	img->p[k]=(BYTE **)malloc(sizeof(BYTE)*img->h*img->w);
	if(img->p[k]==NULL){
//	  puts("# メモリ領域確保失敗");
	  exit(EXIT_FAILURE);
	}
	for(i=0;i<img->h;i++){
	  img->p[k][i]=(BYTE *)malloc(sizeof(BYTE)*img->w);
	  if(img->p[k][i]==NULL){
//		puts("# メモリ領域確保失敗");
		exit(EXIT_FAILURE);
	  }
	}
  }

  /*画像データを読み込んで画像用配列に代入する*/
  for(y=0;y<img->h;y++){
	for(x=0;x<img->w;x++){
	  for(c=0;c<3;c++){
		img->p[c][y][x]=(BYTE)fgetc(fp);
	  }
	}
  }
  fclose(fp);

  return(img);
}


int PPMWrite(PPMStruct *img, char *fname)
{
  FILE *fp;  /* ファイルポインタ */
  int c,x,y;   /* ループ変数 */

  switch(img->t){
  case 6:
	if((fp=fopen(fname,"wb"))==NULL){  /* 出力ファイルのオープン */
//	  puts("# ファイルオープン失敗");
	  exit(EXIT_FAILURE);
	}
	fputs("P6\n",fp); /* ファイル識別子"P6"を先頭に出力する */
	fputs("# Created by pnmio.lib\n",fp);  /* #で始まるコメント行（省略可能） */
	fprintf(fp,"%ld %ld\n",img->w,img->h);  /* 画像の横幅・縦幅の出力 */
	fprintf(fp,"%ld\n",img->m);  /* 最大階調値の出力 */
	for(y=0;y<img->h;y++){  /* 画像データの出力 */
	  for(x=0;x<img->w;x++){
		for(c=0;c<3;c++){
		  fputc(img->p[c][y][x],fp);
		}
	  }
	}
	fclose(fp);
	break;

  default:
//	puts("# ファイルタイプの設定が不適切です");
	return(1);
  }

  return(0);
}