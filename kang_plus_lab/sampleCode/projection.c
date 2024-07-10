#include <stdio.h> 
#include "pnmio.c"
#include <math.h>
#include <stdlib.h>

#define PI 3.14159265358979
#define ptheta 11.48
#define thetak  ptheta * PI / 180
#define cmax 200
#define e    0.0000000000001
#define l 3
#define swap(type, x, y) do { type t = x; x = y; y = t; } while (0)

double f(double x);
void rgb2lab(int nlr, int nlg, int nlb, double *ls, double *as, double *bs);
void lab2lch(double as, double bs, double *chroma, double *hangle);
void lch2lab(double *as, double *bs, double chroma, double hangle);
void lab2drgb(double *lr, double *lg, double *lb, double ls, double as, double bs);
void lab2rgb(int *nlr, int *nlg, int *nlb, double ls, double as, double bs);
double d(double x);
double areajudge(double ls, double as, double bs);
void img0rgb2lab(PPMStruct *img_in, double ***a);
void quick(int a[], int left, int right);
void sejun2(double ***a, int b[], int w, int h, int x);
void masterselect2(int *a, int b[], double ***c, int w, int h);
double distribution_range(int w, int h);
double gaussian_pairing(int i, int h, double r);
void matrix_e(double ***a, double ***hpq, double c[][3][3], double lsigma, double wsigma, int m, int d[], int w, int h);
double kphi3(double a, double b, double c, double d);
void kprint(int a, double phi, double u[]);
void s3ij(double ***a, double ***b, double phi[], int m, int c[], int w, int h);
void rotation_ldd(PPMStruct *img_out, double ***a, double ***b, double ***s, int w, int h);
double lddout(double ***a, double ***b, double ***hpq, double lsigma, int w, int h);
double lddin(double ***a, double ***hpq, double ***s, double lsigma, int w, int h);
void projection(PPMStruct *img_out, double ***a, double ***b, int w, int h);
double rand1(void);
double rand_gauss(void);
double ***DBLCreate_3(int k,int w,int h);
void DBLFree_3(double ***a,int k, int h);

int main(void)
{
	PPMStruct *img_in;
	PPMStruct *img_out;
	char iname[MAX_FILENAME], fname[MAX_FILENAME];

	int w, h, i, kaiten;
	double ***a, ***b;
	double doc;


	//sprintf(iname, "Parrots_k");
	//sprintf(iname, "smallbird_ldd1_h2_select_2");//ball_pool
	//sprintf(iname, "chart26_double_100_ldd4_h2_2_5_10.00_5.0");
	//sprintf(iname, "227_ldd4c_h2_1_19_2.00_5.0");//Parrots_k_ldd4_h2_2_19_10.00_5.0");
	//sprintf(iname, "Parrots_k_ldd1_h2_select_2");//COLOR226_ldd1_h2_select_2");//

	//sprintf(iname, "Parrots_k_ldd4_h2_2_19_2.00_2.0");//te3_Parrots_ldd4_h2_2_5_10.00_5.0");
	sprintf(iname, "RQV74_ldd4_h2_1_50_1.00_1.0");//
	//sprintf(iname, "Japan_ldd4_h2_2_20_4kara_2.00_2.0");
	//sprintf(iname, "pastel");//home");//

	sprintf(fname, "%s.ppm", iname);
	img_in  = PPMRead(fname);
   	img_out = PPMCreate(img_in->w, img_in->h, img_in->m);
	printf("%s.ppm\n", iname);
	puts("a");

	w = img_in->w;
	h = img_in->h;

	a = DBLCreate_3(l, w, h);
	b = DBLCreate_3(l, w, h);

    img0rgb2lab(img_in, a);

	projection(img_out, a, b, w, h);
	//sprintf(fname, "2_%s_%.1f.ppm", iname,doc);
	sprintf(fname, "2_%s.ppm", iname);
	PPMWrite(img_out,fname);

	PPMFree(img_in);
	PPMFree(img_out);
	DBLFree_3(a, l, h);
	DBLFree_3(b, l, h);

	return 0;
}

double f(double x){
	if (x > 0.008856) return pow(x, 1./3);
	else              return (7.78 * x + 16./116);
}
void rgb2lab(int nlr, int nlg, int nlb, double *ls, double *as, double *bs){
	double lr, lg, lb, x, y, z;
	double xn, yn, zn, ax, ay, az;

	lr = pow(nlr / 255., 2.2);
	lg = pow(nlg / 255., 2.2);
	lb = pow(nlb / 255., 2.2);

	x = 0.4124 * lr + 0.3576 * lg + 0.1805 * lb;
	y = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
	z = 0.0193 * lr + 0.1192 * lg + 0.9505 * lb;

	xn = 0.4124 + 0.3576 + 0.1805;
	yn = 0.2126 + 0.7152 + 0.0722;
	zn = 0.0193 + 0.1192 + 0.9505;

	ax = x / xn;
	ay = y / yn;
	az = z / zn;

	*ls = 116 * f(ay) - 16;
	*as = 500 * (f(ax) - f(ay));
	*bs = 200 * (f(ay) - f(az));
}
void lab2lch(double as, double bs, double *chroma, double *hangle){
	double range;

	range = as * as + bs * bs;
	*chroma = pow(range, 0.5);
	*hangle = atan2(bs, as);
}
void lch2lab(double *as, double *bs, double chroma, double hangle){
	*as = chroma * cos(hangle);
	*bs = chroma * sin(hangle);
}
void lab2drgb(double *lr, double *lg, double *lb, double ls, double as, double bs){
	double x, y, z;
	double xn, yn, zn, tx, ty, tz;

	ty = (16 + ls) / 116;
	tx = ty + as / 500;
	tz = ty - bs / 200;

	ty = pow(ty, 3);

	xn = 0.4124 + 0.3576 + 0.1805;
	yn = 0.2126 + 0.7152 + 0.0722;
	zn = 0.0193 + 0.1192 + 0.9505;

	if ( ty > 0.008856) y = ty * yn;
	else 		   y = (yn * ls) / 903.29;

	if ( tx > 0.20690) x = pow(tx, 3) * xn;
	else 		   x = xn * (116 * tx - 16) / 903.29;

	if ( tz > 0.20690) z = pow(tz, 3) * zn;
	else 		   z = zn * (116 * tz - 16) / 903.29;

	*lr =   3.2406 * x - 1.5372 * y - 0.4986 * z;//
	*lg = - 0.9689 * x + 1.8758 * y + 0.0415 * z;//
	*lb =   0.0557 * x - 0.2040 * y + 1.0570 * z;//

}
void lab2rgb(int *nlr, int *nlg, int *nlb, double ls, double as, double bs){
	double lr, lg, lb, x, y, z;
	double xn, yn, zn, tx, ty, tz;

	ty = (16 + ls) / 116;
	tx = ty + as / 500;
	tz = ty - bs / 200;

	ty = pow(ty, 3);

	xn = 0.4124 + 0.3576 + 0.1805;
	yn = 0.2126 + 0.7152 + 0.0722;
	zn = 0.0193 + 0.1192 + 0.9505;

	if ( ty > 0.008856) y = ty * yn;
	else 		    y = (yn * ls) / 903.29;

	if ( tx > 0.20690) x = pow(tx, 3) * xn;
	else 		   x = xn * (116 * tx - 16) / 903.29;

	if ( tz > 0.20690) z = pow(tz, 3) * zn;
	else 		   z = zn * (116 * tz - 16) / 903.29;

	lr =   3.2406 * x - 1.5372 * y - 0.4986 * z;//
	lg = - 0.9689 * x + 1.8758 * y + 0.0415 * z;//
	lb =   0.0557 * x - 0.2040 * y + 1.0570 * z;//

	*nlr = 255 * pow(lr, 1 / 2.2);
	*nlg = 255 * pow(lg, 1 / 2.2);
	*nlb = 255 * pow(lb, 1 / 2.2);
}
double d(double x){
	if (0 <= x && x <= 1) return 1;
	else                return 0;
}
double areajudge(double ls, double as, double bs){
	double chroma, hangle, r, g, b;
	double cin, cmid, cout, djudge;

	lab2lch(as, bs, &chroma, &hangle);
	cin = 0;
	cout = cmax;

	do{
		cmid = (cin + cout) / 2;
		lch2lab(&as, &bs, cmid, hangle);
		lab2drgb(&r, &g, &b, ls, as, bs);
		djudge = d(r) * d(g) * d(b);

		if(djudge == 1)
			cin = cmid;
		else
			cout = cmid;

	}while((cout - cin) > e);

	return cin;
}
void img0rgb2lab(PPMStruct *img_in, double ***a){
	int i, j, rr, gg, bb;
	double ls, as, bs;

	for(i = 0; i < img_in->h; i++){
		for(j = 0; j < img_in->w; j++){ 
			rr = img_in->p[0][i][j];
			gg = img_in->p[1][i][j];
			bb = img_in->p[2][i][j];

			rgb2lab(rr, gg, bb, &ls, &as, &bs);

			a[0][i][j] = ls;
			a[1][i][j] = as;
			a[2][i][j] = bs;
		}
	}

}
void quick(int a[], int left, int right){
   int pl = left;
   int pr = right;
   int x = a[(pl + pr) / 2];

   do{
      while (a[pl] < x) pl++;
      while (a[pr] > x) pr--;
      if (pl <= pr){ 
         swap(int, a[pl], a[pr]);
         pl++;
         pr--;
      }
   } while (pl <= pr);
   if (left < pr)  quick(a, left, pr);
   if (pl < right) quick(a, pl, right);
}
void sejun2(double ***a, int light[], int w, int h, int x){
    int i, j;
    int count = 0;
    int s[200000] = {0};

    for(i = 0; i < h; i++){
		for(j = 0; j < w; j++){
			if(a[0][i][j] != 100 && a[0][i][j] != 0){
				s[count] = a[0][i][j];
				count++;
			}
		}
	}

    quick(s, 0, count - 1);
    for(i = 0; i < x; i++) light[x - i - 1] = s[(i + 1) * count / (x + 1)];
}
void masterselect2(int *a, int b[], double ***c, int w, int h){
    int i, x, y;

    /*printf("代表明度の数を決めて下さい(1~100)\n");
    scanf("%d", &x);*/
	x = 100;
    *a = x;
	if(x == 100){
		for(i = 0; i < x; i++){
        	b[i] = i + 1;
		}
	}
    else if(x == 50){
		for(i = 0; i < x; i++){
        	b[i] = i * 2 + 1;
		}
	}
    else if(x == 20){
		for(i = 0; i < x; i++){
        	b[i] = i * 5 + 3;
		}
	}
    else if(x == 10){
		for(i = 0; i < x; i++){
        	b[i] = i * 10 + 5;
		}
	}
    else if(x == 5){
		for(i = 0; i < x; i++){
        	b[i] = i * 20 + 10;
		}
	}
    else{
		printf("代表明度を決めて下さい(1~99or0)\n");
    	for(i = 0; i < x; i++){
        	printf("%d：",i + 1);
        	scanf("%d", &y);
            if(y == 0){
                sejun2(c, b, w, h, x);
                printf("累積による代表明度\n");
                for(i = 0; i < x; i++) printf("%d：%d\n",i + 1, b[i]);
                break;
            }
            b[i] = y;
		}
    }
    quick(b, 0, x - 1);
    printf("osimai\n");
}
double distribution_range(int w, int h){
	int min;
	double r;

	if(w > h) min = h;
	else min = w;

	r = sqrt(2. * min) * 2. / PI / 3.;
	return r;
}
double gaussian_pairing(int i, int h, double r){
	int p;

	p = i + r * rand_gauss();		
	if(p < 0) p = - p;
	if(p >= h) p = 2 * h - p - 2;

	return p;
}
void matrix_e(double ***a, double ***hpq, double c[][3][3], double lsigma, double wsigma, int m, int d[], int w, int h){
	int i, j, p, q, s, t, u;
	double r, ldw, lav;
	double lbw[101] = {0};
	double ***x;

	x = DBLCreate_3(l, w, h);
	r = distribution_range(w, h);

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			p = gaussian_pairing(i, h, r);
			q = gaussian_pairing(j, w, r);
			hpq[0][i][j] = p;
			hpq[1][i][j] = q;

			x[0][i][j] = a[0][i][j] - a[0][p][q];
			x[1][i][j] = a[1][i][j] - a[1][p][q];  
			x[2][i][j] = a[2][i][j] - a[2][p][q];  

			ldw = exp(x[0][i][j] * x[0][i][j] / (-2. * lsigma * lsigma));
			lav = (a[0][i][j] + a[0][p][q]) / 2;
			for(u = 0; u < m; u++){
				lbw[u] = exp((lav - d[u]) * (lav - d[u])  / (-2 * wsigma * wsigma));
			}
			for(s = 0; s < 3; s++){
       			for(t = 0; t < 3; t++){
					for(u = 0; u < m; u++){
						c[u][s][t] += ldw * lbw[u] * x[s][i][j] * x[t][i][j];// 
					} 
				}
			}
		}
	}

	DBLFree_3(x, l, h);
}
double kphi3(double a, double b, double c, double d){
	double range, z, alpha, zphi, ophi, x, y, phi;

	range = (c - b);
	z = 2 * a;
	alpha = atan2(z, range);

	zphi = 0 - alpha * 0.5;
	ophi = PI - alpha * 0.5;

	x = cos(zphi) * cos(d) + sin(zphi) * sin(d);
	y = cos(ophi) * cos(d) + sin(ophi) * sin(d);

	if(x > y)
		phi = zphi;
	else
		phi = ophi;

	if(fabs(d - phi) < PI)
		return phi;
	else if( (d -phi) < 0){
		phi -= 2 * PI;
		return phi;
	}
	else{
		phi += 2 * PI;
		return phi;
	}
}
void kprint(int a, double phi, double u[]){
	double k;
	k = phi * 180 / PI;
	printf("k%d: %.1f, phi%d: %f\n", a, k,  a, phi);
	printf("b: u%d[0] =  %f, u%d[1] =  %f, u%d[2] =  %f\n", a, u[0], a, u[1], a, u[2]);
}
void s3ij(double ***a, double ***b, double phi[], int m, int c[], int w, int h){
    int i, j, k, ldt;
	double shi;

    for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			ldt = (int)a[0][i][j]; 
			if(a[0][i][j] <= c[0]){
				shi = phi[1];
			}
			for(k = 0; k < m - 1; k++){
				 if(a[0][i][j] == c[k]){
				   shi = phi[k + 1];
			    }
			    if(c[k] < a[0][i][j] && a[0][i][j] < c[k + 1]){
				    shi = (a[0][i][j] - c[k]) * (phi[k + 2] - phi[k + 1]) / (c[k + 1] - c[k]) + phi[k + 1];
			    }
            }
            if(a[0][i][j] >= c[m - 1]){
                shi = phi[m];
			}
            b[0][i][j] = shi;
			b[1][i][j] = cos(shi);
			b[2][i][j] = sin(shi); 
		}
	}
}
void rotation_ldd(PPMStruct *img_out, double ***a, double ***b, double ***s, int w, int h){
    int i, j, rr, gg, bb;
	double rotate;

    for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			rotate = thetak - s[0][i][j] + PI;

			b[0][i][j] = a[0][i][j];
			b[1][i][j] = a[1][i][j] * cos(rotate) - a[2][i][j] * sin(rotate);
			b[2][i][j] = a[1][i][j] * sin(rotate) + a[2][i][j] * cos(rotate);

		    lab2rgb(&rr, &gg, &bb, b[0][i][j], b[1][i][j], b[2][i][j]);

			img_out->p[0][i][j] = rr;
			img_out->p[1][i][j] = gg;
			img_out->p[2][i][j] = bb;
		}
	}
}
double lddout(double ***a, double ***b, double ***hpq, double lsigma, int w, int h){
	int i, j, p, q;
	double ltheta = 0;
    double wall = 0;
	double xtu, noa, nob;
	double ***xd, ***xa, ***wk;

	xd = DBLCreate_3(l, w, h);
	xa = DBLCreate_3(l, w, h);
	wk= DBLCreate_3(l, w, h);

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			p = hpq[0][i][j];
			q = hpq[1][i][j];

			xd[0][i][j] = b[0][i][j] - b[0][p][q];
			xd[1][i][j] = b[1][i][j] - b[1][p][q];  
			xd[2][i][j] = b[2][i][j] - b[2][p][q];

            xa[0][i][j] = a[0][i][j] - a[0][p][q];
			xa[1][i][j] = a[1][i][j] - a[1][p][q];  
			xa[2][i][j] = a[2][i][j] - a[2][p][q];  

            nob = xd[1][i][j] * xd[1][i][j] + xd[2][i][j] * xd[2][i][j];
            noa = xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];

			xtu = xd[1][i][j] * cos(thetak) + xd[2][i][j] * sin(thetak);
			wk[0][i][j] = exp(xd[0][i][j] * xd[0][i][j] / (-2 * lsigma * lsigma));
            if(nob == 0){
                ltheta += 0;
            }
            else{
                ltheta += wk[0][i][j] * xtu * xtu * noa / nob;  
            }
            
            wall += wk[0][i][j];
		}
	}
	
	//ltheta = ltheta / (h * w);
	ltheta = ltheta / wall;
	printf("ltheta: %f\n", ltheta);

    DBLFree_3(xd, l, h);
    DBLFree_3(xa, l, h);
	DBLFree_3(wk, l, h);
    
    return ltheta;
}
double lddin(double ***a, double ***hpq, double ***s, double lsigma, int w, int h){
	int i, j, p, q;
	double ltheta = 0;
    double wall = 0;
	double ***x, ***xtu, ***wk;

    x = DBLCreate_3(l, w, h);
	xtu = DBLCreate_3(l, w, h);
	wk= DBLCreate_3(l, w, h);

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
            p = hpq[0][i][j];
			q = hpq[1][i][j];
            x[0][i][j] = a[0][i][j] - a[0][p][q];
			x[1][i][j] = a[1][i][j] - a[1][p][q];  
			x[2][i][j] = a[2][i][j] - a[2][p][q];

			xtu[0][i][j] = x[1][i][j] * s[1][i][j] + x[2][i][j] * s[2][i][j];
			wk[0][i][j] = exp(x[0][i][j] * x[0][i][j] / (-2 * lsigma * lsigma));
            ltheta += wk[0][i][j] * xtu[0][i][j] * xtu[0][i][j];
            wall += wk[0][i][j];
		}
	}

	ltheta = ltheta / wall;
	printf("ltheta: %f\n", ltheta);

    DBLFree_3(x, l, h);
    DBLFree_3(xtu, l, h);
	DBLFree_3(wk, l, h);
    
    return ltheta;
}
void projection(PPMStruct *img_out, double ***a, double ***b, int w, int h){
    int i, j, rr, gg, bb;
	double rotate, atu, cborder2, chroma, hangle;

    for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			rotate = thetak + PI;

			atu = a[1][i][j] * cos(rotate) + a[2][i][j] * sin(rotate);

			b[0][i][j] = a[0][i][j];
			b[1][i][j] = a[1][i][j] - atu * cos(rotate);
			b[2][i][j] = a[2][i][j] - atu * sin(rotate);

			cborder2 = areajudge(b[0][i][j], b[1][i][j], b[2][i][j]);
			lab2lch(b[1][i][j], b[2][i][j], &chroma, &hangle);
			if(chroma>cborder2) lch2lab(&b[1][i][j], &b[2][i][j], cborder2, hangle);

		    lab2rgb(&rr, &gg, &bb, b[0][i][j], b[1][i][j], b[2][i][j]);

			img_out->p[0][i][j] = rr;
			img_out->p[1][i][j] = gg;
			img_out->p[2][i][j] = bb;
		}
	}
}
double rand1(void){
	return((double)rand()/RAND_MAX);
}
double rand_gauss(void){
	static int sw=0;
	static double t,u;
	double ww,mm;
	if(sw==0){
		sw=1;
		t=sqrt(-2.*log(1.-rand1()));
		u=2.*PI*rand1();
		ww=t*cos(u);
		return(ww);
	}else{
		sw=0;
		mm=t*sin(u);
		return(mm);
	}
}
double ***DBLCreate_3(int k,int w,int h){
        int i,j;
        double ***a;
        a=(double ***)malloc(sizeof(double **)*k);
        for (i=0;i<k;i++){
                a[i]=(double **)malloc(sizeof(double *)*h);
                for (j=0;j<h;j++){
                        a[i][j]=(double *)malloc(sizeof(double)*w);
                }
        }
        return(a);
}
void DBLFree_3(double ***a,int k, int h){
        int i,j;
        for(i=0;i<k;i++){
                for(j=0;j<h;j++){
                        free(a[i][j]);
                }
                free(a[i]);
        }
        free(a);
}