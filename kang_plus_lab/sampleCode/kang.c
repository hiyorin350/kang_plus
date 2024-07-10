#include <stdio.h> 
#include "pnmio.c"
#include <math.h>
#include <stdlib.h>

#define PI 3.14159265358979
#define dtheta 8.11
#define ptheta 11.48
#define thetak  ptheta * PI / 180
#define cmax 200
#define e    0.0000000000001
#define l 3
#define ptau3siki 12
#define ptau2siki 8
#define ftau3siki 5
#define ftau2siki 15 //change
#define perm 0.1

double f(double x);
void rgb2lab(int nlr, int nlg, int nlb, double *ls, double *as, double *bs);
void lab2lch(double as, double bs, double *chroma, double *hangle);
void lch2lab(double *as, double *bs, double chroma, double hangle);
void lab2drgb(double *lr, double *lg, double *lb, double ls, double as, double bs);
void lab2rgb(int *nlr, int *nlg, int *nlb, double ls, double as, double bs);
double d(double x);
double areajudge(double ls, double as, double bs);
void img0rgb2lab(PPMStruct *img_in, double ***a);
double distribution_range(int w, int h);
double gaussian_pairing(int i, int h, double r);
double matrix_a(double ***a, double ***b, double ***hpq, double sigma, int w, int h);
double kphi3(double a, double b, double c, double d);
void rotation_c2(PPMStruct *img_out, double ***a, double ***b, double ***c, double ***hpq, double phi, double lsigma, int *count, int w, int h);
void over_black(PPMStruct *img_out, double ***a, double ***c, double phi, int w, int h);
double ldd_abridged(double ***a, double ***b, double ***hpq, double lsigma, int i, int j);
void ldd_in1(double ***x, double u[], double lsigma, int w, int h);
double ldd_out_ab(double ***a, double ***b, double ***hpq, double lsigma, int w, int h);
double ldd_souwarate(double ***a, double ***b, double ***hpq, double lsigma, int w, int h);
double ldd_h2_souwarate(double ***a, double ***b, double ***hpq, double lsigma, int w, int h);
double ldd_h2_norm_souwarate(double ***a, double ***b, double ***hpq, double ***norm, double lsigma, int w, int h);
double ldd_h2_norm_souwarate_torn(double ***a, double ***b, double ***hpq, double ***norm, double *l3_to, double *l3_no, int *c, int *c_to, double lsigma, int w, int h);
double l1_h2_norm_souwarate(double ***a, double ***b, double ***hpq, int w, int h);
double l2_h2_relative_error(double ***a, double ***b, double ***hpq, char *iname, int w, int h);
double ldd_h2_abnorm_souwarate(double ***a, double ***b, double ***hpq, double ***norm, double lsigma, int w, int h);
void projection0(double ***a, double ***b, int w, int h);
void projection_c(double ***a, double ***b, int w, int h);
void escore(PPMStruct *img_ref_p, PPMStruct *img_ref_f, double ***a, double ***b, double ***hpq, double *ccpr, double *ccfr, double *escore, int w, int h);
void escore_is(PPMStruct *img_ref_p, PPMStruct *img_ref_f, double ***a, double ***b, double ***hpq, double *ccpr, double *ccfr, double *escore, int w, int h);
void w2_xnorm(PPMStruct *img_ref_w2, double ***a, int w, int h);
double ph_v3(double x, double y);
double ph_v4(double x, double y);
void v_a(double ***a, double ***b, double ***hpq, double *v_1, double *v_2, double *v_3, double *v_4, double esigma, char *iname, int w, int h);
void v_b(double ***a, double ***b, double ***hpq, double *v_1, double *v_2, double *v_3, double *v_4, double esigma, char *iname, int w, int h);
void v_c(double ***a, double ***b, double ***hpq, double *v_1, double *v_2, double *v_3, double *v_4, double esigma, char *iname, int w, int h);
//void random_pair(double ***hpq, int w, int h);
//int getRandomNumber(int min, int max);
double rand1(void);
double rand_gauss(void);
double ***DBLCreate_3(int k,int w,int h);
void DBLFree_3(double ***a,int k, int h);

int main(void)
{

	PPMStruct *img_in;
	PPMStruct *img_out;
    PPMStruct *img_ref_p;
    PPMStruct *img_ref_f;

	char iname[MAX_FILENAME], fname[MAX_FILENAME];

	int w, h, i, j, count_a, count_b, select;
	double phi_a, phi_b, k_a, k_b, lsigma, esigma, ld_a, ld_b, ld_c, ldd_a, ldd_b, ldd_c, lddd_a, lddd_b, lddd_c;
    double ccpr_a, ccpr_b, ccpr_c, ccfr_a, ccfr_b, ccfr_c, escore_a, escore_b, escore_c, rotate;
	double l1_c, l2_c, yx_abs, rel_er, yx_pm;
	double lddd_to_a, lddd_to_b, lddd_to_c, lddd_no_a, lddd_no_b, lddd_no_c;
    double v_1a, v_1b, v_1c, v_2a, v_2b, v_2c, v_3a, v_3b, v_3c, v_4a, v_4b, v_4c;
	int cn_a, cn_b, cn_c, cnto_a, cnto_b, cnto_c;
	double ***a, ***x, ***b, ***c, ***hpq, ***rpair, ***norm_a, ***norm_b, ***norm_c;

	select = 2;

	sprintf(iname, "Parrots_k");//RQV74");//Japan");//
	sprintf(fname, "%s.ppm", iname);
	printf("%s.ppm\n", iname);
	img_in  = PPMRead(fname);
    img_out = PPMCreate(img_in->w, img_in->h, img_in->m);
    img_ref_p = PPMCreate(img_in->w, img_in->h, img_in->m);
   	img_ref_f = PPMCreate(img_in->w, img_in->h, img_in->m);

	w = img_in->w;
	h = img_in->h;

	lsigma = 5;
	esigma = 5;
	printf("perm=%.2f\n", perm);

	a = DBLCreate_3(l, w, h);
	x = DBLCreate_3(l, w, h);
	b = DBLCreate_3(l, w, h);
    c = DBLCreate_3(l, w, h);
	hpq = DBLCreate_3(l, w, h);
    rpair = DBLCreate_3(l, w, h);
    norm_a = DBLCreate_3(4, w, h);
    norm_b = DBLCreate_3(4, w, h);
	norm_c = DBLCreate_3(4, w, h);

	img0rgb2lab(img_in, a);

	phi_a =  matrix_a(a, x, hpq, lsigma, w, h);
	phi_b = phi_a + PI;
	k_a = phi_a * 180. / PI;
	k_b = phi_b * 180. / PI;
    //random_pair(rpair, w, h);

	double u[3] = {0, cos(phi_a), sin(phi_a)};
	printf("k: %f, phi: %f\n", k_a,  phi_a);
	printf("b: u[0] =  %f, u[1] =  %f, u[2] =  %f\n", u[0], u[1], u[2]);

    rotation_c2(img_out, a, b, c, hpq, phi_a, esigma, &count_a, w, h);
	ld_a = ldd_out_ab(a, c, hpq, esigma, w, h);
	ldd_a = ldd_h2_souwarate(a, c, hpq, esigma, w, h);
	lddd_a = ldd_h2_norm_souwarate_torn(a, c, hpq, norm_a, &lddd_to_a, &lddd_no_a, &cn_a, &cnto_a, esigma, w, h);
    v_a(a, c, hpq, &v_1a, &v_2a, &v_3a, &v_4a, esigma, iname, w, h);
	escore(img_ref_p, img_ref_f, a, c, hpq, &ccpr_a, &ccfr_a, &escore_a, w, h);
	sprintf(fname, "%s_ldd1_h2_select_1.ppm", iname);
	PPMWrite(img_out,fname);

	sprintf(fname, "%s_ldd1_escore_ccpr_m1rg_%.1f_3tau%d_2tau%d.ppm", iname, esigma, ptau3siki, ptau2siki);
	PPMWrite(img_ref_p,fname);
	sprintf(fname, "%s_ldd1_escore_ccfr_m1rg_%.1f_3tau%d_2tau%d.ppm", iname, esigma, ftau3siki, ftau2siki);
	PPMWrite(img_ref_f,fname);

	over_black(img_out, a, c, phi_a, w, h);
	sprintf(fname, "%s_overblack_a_ldd1.ppm", iname);
	PPMWrite(img_out,fname);

    rotation_c2(img_out, a, b, c, hpq, phi_b, esigma, &count_b, w, h);
	ld_b = ldd_out_ab(a, c, hpq, esigma, w, h);
	ldd_b = ldd_h2_souwarate(a, c, hpq, esigma, w, h);
	lddd_b = ldd_h2_norm_souwarate_torn(a, c, hpq, norm_b, &lddd_to_b, &lddd_no_b, &cn_b, &cnto_b, esigma, w, h);
    v_b(a, c, hpq, &v_1b, &v_2b, &v_3b, &v_4b, esigma, iname, w, h);
	escore(img_ref_p, img_ref_f, a, c, hpq, &ccpr_b, &ccfr_b, &escore_b, w, h);
	sprintf(fname, "%s_ldd1_h2_select_2.ppm", iname);
	PPMWrite(img_out,fname);	

	sprintf(fname, "%s_ldd1_escore_ccpr_m2rg_%.1f_3tau%d_2tau%d.ppm", iname, esigma, ptau3siki, ptau2siki);
	PPMWrite(img_ref_p,fname);
	sprintf(fname, "%s_ldd1_escore_ccfr_m2rg_%.1f_3tau%d_2tau%d.ppm", iname, esigma, ftau3siki, ftau2siki);
	PPMWrite(img_ref_f,fname);

	over_black(img_out, a, c, phi_b, w, h);
	sprintf(fname, "%s_overblack_b_ldd1.ppm", iname);
	PPMWrite(img_out,fname);

	ld_c = ldd_out_ab(a, b, hpq, esigma, w, h);
	ldd_c = ldd_h2_souwarate(a, b, hpq, esigma, w, h);
	lddd_c = ldd_h2_norm_souwarate_torn(a, b, hpq, norm_c, &lddd_to_c, &lddd_no_c, &cn_c, &cnto_c, esigma, w, h);
	l1_c = l1_h2_norm_souwarate(a, b, hpq, w, h);
	l2_c = l2_h2_relative_error(a, b, hpq, iname, w, h);
    v_c(a, b, hpq, &v_1c, &v_2c, &v_3c, &v_4c, esigma, iname, w, h);
	escore_is(img_ref_p, img_ref_f, a, b, hpq, &ccpr_c, &ccfr_c, &escore_c, w, h);

	sprintf(fname, "%s_ldd1_escore_ccpr_m0rg_%.1f_3tau%d_2tau%d.ppm", iname, esigma, ptau3siki, ptau2siki);
	PPMWrite(img_ref_p,fname);
	sprintf(fname, "%s_ldd1_escore_ccfr_m0rg_%.1f_3tau%d_2tau%d.ppm", iname, esigma, ftau3siki, ftau2siki);
	PPMWrite(img_ref_f,fname);

	if(count_b >= count_a) select = 1;
	
	ldd_in1(x, u, esigma, w, h);

	k_a = ptheta - k_a + 180;	
	k_b = ptheta - k_b + 180;	
	
    FILE *fp;

	sprintf(fname, "kang_h2_hyouka1201v4.csv");
	//sprintf(fname, "%s_ldd4_h2_escore0904.csv", iname);
    fp = fopen(fname, "a");
	
	if(fp == NULL){
		printf("dame\n");
		return -1;
	}

	fprintf(fp, "%s,%d,%d,%d,%d,%d,a,%d,%f,%f,%f,%f,%f,%f,b,%d,%f,%f,%f,%f,%f,%f,c,%f,%f,%f,%f,%f,%f,%f,%f\n",iname,select,ptau3siki,ptau2siki,ftau3siki,ftau2siki,count_a,ld_a,ldd_a,lddd_a,ccpr_a,ccfr_a,escore_a,count_b,ld_b,ldd_b,lddd_b,ccpr_b,ccfr_b,escore_b,ld_c,ldd_c,lddd_c,ccpr_c,ccfr_c,escore_c,l1_c,l2_c);	
	fprintf(fp, "%s,lddd_a,%f,lddd_to_a,%f,lddd_no_a,%f,lddd_b,%f,lddd_to_b,%f,lddd_no_b,%f,lddd_c,%f,lddd_to_c,%f,lddd_no_c,%f\n",iname,lddd_a,lddd_to_a,lddd_no_a,lddd_b,lddd_to_b,lddd_no_b,lddd_c,lddd_to_c,lddd_no_c);
	fprintf(fp, "%s,cn_a,%d,cnto_a,%d,cn_b,%d,cnto_b,%d,cn_c,%d,cnto_c,%d\n",iname,cn_a,cnto_a,cn_b,cnto_b,cn_c,cnto_c);	
    fprintf(fp, "%s,%f,v_1a,%f,v_2a,%f,v_3a,%f,v_4a,%f,v_1b,%f,v_2b,%f,v_3b,%f,v_4b,%f,v_1c,%f,v_2c,%f,v_3c,%f,v_4c,%f\n",iname,perm,v_1a,v_2a,v_3a,v_4a,v_1b,v_2b,v_3b,v_4b,v_1c,v_2c,v_3c,v_4c);	
	//fprintf(fp, "%s,ltheta:,a,%f,b,%f,c,%f\n", iname, ltheta2_a, ltheta2_b, ltheta2_c);
	fclose(fp);


	sprintf(fname, "%s_ldd1_h2_norm_c_%.1f_re.csv", iname, esigma);
    fp = fopen(fname, "a");
	
	if(fp == NULL){
		printf("dame\n");
		return -1;
	}

	fprintf(fp, "i,j,xnorm,ynorm,w2_x,w2_y,||y-x||,rel_er,y-x\n");
	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			yx_pm = norm_c[1][i][j] - norm_c[0][i][j];
			yx_abs = fabs(yx_pm);
			if(norm_c[0][i][j] != 0){
				rel_er = yx_abs / norm_c[0][i][j];				
			}
			else{
				rel_er = 0;
			}

        	fprintf(fp, "%d,%d,%f,%f,%f,%f,%f,%f,%f\n",i,j,norm_c[0][i][j],norm_c[1][i][j],norm_c[2][i][j],norm_c[3][i][j],yx_abs,rel_er,yx_pm);
		}
	}

	fclose(fp);





	sprintf(fname, "%s_ldd1_h2_1_kakudo.csv", iname);
    fp = fopen(fname, "a");
	
	if(fp == NULL){
		printf("dame\n");
		return -1;
	}
	fprintf(fp, "%f",esigma);

    rotate = thetak - phi_a + PI;
    rotate = rotate * 180. / PI;
    fprintf(fp, ",%f",rotate);

    fprintf(fp, "\n");
	fclose(fp);


	sprintf(fname, "%s_ldd1_h2_2_kakudo.csv", iname);
    fp = fopen(fname, "a");
	
	if(fp == NULL){
		printf("dame\n");
		return -1;
	}
	fprintf(fp, "%f",esigma);

    rotate = thetak - phi_b + PI;
    rotate = rotate * 180. / PI;
    fprintf(fp, ",%f",rotate);

    fprintf(fp, "\n");
	fclose(fp);


	PPMFree(img_in);
	PPMFree(img_out);
	PPMFree(img_ref_p);
	PPMFree(img_ref_f);
	DBLFree_3(a, l, h);
	DBLFree_3(x, l, h);
	DBLFree_3(b, l, h);
    DBLFree_3(c, l, h);
	DBLFree_3(hpq, l, h);
    DBLFree_3(rpair, l, h);
	DBLFree_3(norm_a, 4, h);
	DBLFree_3(norm_b, 4, h);
	DBLFree_3(norm_c, 4, h);

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
		cmid = (cin + cout) / 2.;
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
double matrix_a(double ***a, double ***b, double ***hpq, double sigma, int w, int h){
	int i, j, p, q, s, t;
	double r, wij, phi;
	double c[3][3] = {0};

	r = distribution_range(w, h);

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			p = gaussian_pairing(i, h, r);
			q = gaussian_pairing(j, w, r);
			hpq[0][i][j] = p;
			hpq[1][i][j] = q;

			b[0][i][j] = a[0][i][j] - a[0][p][q];
			b[1][i][j] = a[1][i][j] - a[1][p][q];  
			b[2][i][j] = a[2][i][j] - a[2][p][q];  

			wij = exp(b[0][i][j] * b[0][i][j] / (-2. * sigma * sigma));

			for(s = 0; s < 3; s++){
       				for(t = 0; t < 3; t++){
					c[s][t] += b[s][i][j] * b[t][i][j];    
				}
			}
		}
	}
	phi =  kphi3(c[1][2], c[1][1], c[2][2], PI / 2.);
	return phi;    
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
void rotation_c2(PPMStruct *img_out, double ***a, double ***b, double ***c, double ***hpq, double phi,double lsigma, int *count, int w, int h){
	int i, j, rr, gg, bb, p, q;
	double rotate, cborder, chroma, hangle;
    double ldd_b, ldd_c;
    int c_stick = 0;

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			rotate = thetak - phi + PI;			
			b[0][i][j] = a[0][i][j];
			b[1][i][j] = a[1][i][j] * cos(rotate) - a[2][i][j] * sin(rotate);
			b[2][i][j] = a[1][i][j] * sin(rotate) + a[2][i][j] * cos(rotate);

            c[0][i][j] = b[0][i][j];
            c[1][i][j] = b[1][i][j];
            c[2][i][j] = b[2][i][j];
                
            cborder = areajudge(c[0][i][j], c[1][i][j], c[2][i][j]);
            lab2lch(c[1][i][j], c[2][i][j], &chroma, &hangle);
		    if(chroma>cborder){
				lch2lab(&c[1][i][j], &c[2][i][j], cborder, hangle);
				c_stick++;
			} 

            lab2rgb(&rr, &gg, &bb, c[0][i][j], c[1][i][j], c[2][i][j]);
			img_out->p[0][i][j] = rr;
			img_out->p[1][i][j] = gg;
			img_out->p[2][i][j] = bb;
		}
	}

	*count = c_stick;
}
void over_black(PPMStruct *img_out, double ***a, double ***c, double phi, int w, int h){
    int i, j, rr, gg, bb;
	double rotate, cborder, chroma, hangle;

    for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			rotate = thetak - phi + PI;	

			c[0][i][j] = a[0][i][j];
			c[1][i][j] = a[1][i][j] * cos(rotate) - a[2][i][j] * sin(rotate);
			c[2][i][j] = a[1][i][j] * sin(rotate) + a[2][i][j] * cos(rotate);

            cborder = areajudge(c[0][i][j], c[1][i][j], c[2][i][j]);
            lab2lch(c[1][i][j], c[2][i][j], &chroma, &hangle);
		    if(chroma>cborder){
				lch2lab(&c[1][i][j], &c[2][i][j], cborder, hangle);
				c[0][i][j] = 0;
				c[1][i][j] = 0;
				c[2][i][j] = 0;
			} 

            lab2rgb(&rr, &gg, &bb, c[0][i][j], c[1][i][j], c[2][i][j]);

			img_out->p[0][i][j] = rr;
			img_out->p[1][i][j] = gg;
			img_out->p[2][i][j] = bb;
		}
	}
}
double ldd_abridged(double ***a, double ***b, double ***hpq, double lsigma, int i, int j){
    int p, q;
    double xtu, noa, nob, ltheta, wk;
	double xd[3] = {0};
	double xa[3] = {0};   

    p = hpq[0][i][j];
	q = hpq[1][i][j];

	xd[0] = b[0][i][j] - b[0][p][q];
	xd[1] = b[1][i][j] - b[1][p][q];  
	xd[2] = b[2][i][j] - b[2][p][q]; 

    xa[0] = a[0][i][j] - a[0][p][q];
	xa[1] = a[1][i][j] - a[1][p][q];  
	xa[2] = a[2][i][j] - a[2][p][q];

    nob = xd[1] * xd[1] + xd[2] * xd[2];
    noa = xa[1] * xa[1] + xa[2] * xa[2]; 

    xtu = xd[1] * cos(thetak) + xd[2] * sin(thetak);
	wk = exp(xd[0] * xd[0] / (-2 * lsigma * lsigma));
	if(nob == 0){
        ltheta = 0;
    }
    else{
        ltheta = wk * xtu * xtu * noa / nob;  
    }

    return ltheta;
}
void ldd_in1(double ***x, double u[], double lsigma, int w, int h){
	int i, j;
	double ltheta = 0;
	double wall = 0;
	double ***xtu, ***wk;

	xtu = DBLCreate_3(l, w, h);
	wk = DBLCreate_3(l, w, h);

	for(i = 0; i < h; i++){
		for(j = 0; j < w; j++){
			xtu[0][i][j] = x[0][i][j] * u[0] + x[1][i][j] * u[1] + x[2][i][j] *u[2];
			wk[0][i][j] = exp(x[0][i][j] * x[0][i][j] / (-2. * lsigma * lsigma)); 
			ltheta += wk[0][i][j] * xtu[0][i][j] * xtu[0][i][j];
			wall += wk[0][i][j];
		}
	}

	ltheta = ltheta / wall;
	printf("ltheta: %.2f\n", ltheta);

	DBLFree_3(xtu, l, h);
	DBLFree_3(wk, l, h);
}
double ldd_out_ab(double ***a, double ***b, double ***hpq, double lsigma, int w, int h){
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
	printf("ltheta: %.2f\n", ltheta);

    DBLFree_3(xd, l, h);
	DBLFree_3(wk, l, h);
    
    return ltheta;
}
double ldd_souwarate(double ***a, double ***b, double ***hpq, double lsigma, int w, int h){
	int i, j, p, q;
	double lthetax = 0;
	double lthetay = 0;
    double wall = 0;
	double xtu, noa, nob, ltheta;
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

            noa = xa[0][i][j] * xa[0][i][j] + xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];
			noa = sqrt(noa);
			xtu = xd[1][i][j] * cos(thetak) + xd[2][i][j] * sin(thetak);
			wk[0][i][j] = exp(xa[0][i][j] * xa[0][i][j] / (-2 * lsigma * lsigma));
            wk[1][i][j] = exp(xd[0][i][j] * xd[0][i][j] / (-2 * lsigma * lsigma));

			xd[1][i][j] = xd[1][i][j] - xtu * cos(thetak);  
			xd[2][i][j] = xd[2][i][j] - xtu * sin(thetak);

            nob = xd[0][i][j] * xd[0][i][j] + xd[1][i][j] * xd[1][i][j] + xd[2][i][j] * xd[2][i][j];
			nob = sqrt(nob);

		    lthetax += wk[0][i][j] * noa;
            lthetay += wk[1][i][j] * nob;
            
		}
	}

    if(lthetax == 0) ltheta = 0;
    else ltheta = lthetay / lthetax;

	printf("ltheta: %.3f\n", ltheta);

    DBLFree_3(xd, l, h);
	DBLFree_3(wk, l, h);
    
    return ltheta;
}
double ldd_h2_souwarate(double ***a, double ***b, double ***hpq, double lsigma, int w, int h){
	int i, j, p, q;
	double lthetax = 0;
	double lthetay = 0;
    double wall = 0;
	double xtu, noa, nob, ltheta;
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

            noa = xa[0][i][j] * xa[0][i][j] + xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];
			//noa = sqrt(noa);
			xtu = xd[1][i][j] * cos(thetak) + xd[2][i][j] * sin(thetak);
			wk[0][i][j] = exp(xa[0][i][j] * xa[0][i][j] / (-2 * lsigma * lsigma));
            wk[1][i][j] = exp(xd[0][i][j] * xd[0][i][j] / (-2 * lsigma * lsigma));

			xd[1][i][j] = xd[1][i][j] - xtu * cos(thetak);  
			xd[2][i][j] = xd[2][i][j] - xtu * sin(thetak);

            nob = xd[0][i][j] * xd[0][i][j] + xd[1][i][j] * xd[1][i][j] + xd[2][i][j] * xd[2][i][j];
			//nob = sqrt(nob);

		    lthetax += wk[0][i][j] * noa;
            lthetay += wk[1][i][j] * nob;
            
		}
	}

    if(lthetax == 0) ltheta = 0;
    else ltheta = lthetay / lthetax;

	printf("ldd: %.3f\n", ltheta);

    DBLFree_3(xd, l, h);
	DBLFree_3(wk, l, h);
    
    return ltheta;
}
double ldd_h2_norm_souwarate(double ***a, double ***b, double ***hpq, double ***norm, double lsigma, int w, int h){
	int i, j, p, q;
	double lthetax = 0;
	double lthetay = 0;
    double wall = 0;
	double xtu, noa, nob, ltheta;
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

            noa = xa[0][i][j] * xa[0][i][j] + xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];
			noa = sqrt(noa);
			xtu = xd[1][i][j] * cos(thetak) + xd[2][i][j] * sin(thetak);
			wk[0][i][j] = exp(xa[0][i][j] * xa[0][i][j] / (-2 * lsigma * lsigma));
            wk[1][i][j] = exp(xd[0][i][j] * xd[0][i][j] / (-2 * lsigma * lsigma));

			xd[1][i][j] = xd[1][i][j] - xtu * cos(thetak);  
			xd[2][i][j] = xd[2][i][j] - xtu * sin(thetak);

            nob = xd[0][i][j] * xd[0][i][j] + xd[1][i][j] * xd[1][i][j] + xd[2][i][j] * xd[2][i][j];
			nob = sqrt(nob);

		    lthetax += wk[0][i][j] * noa;
            lthetay += wk[1][i][j] * nob;

			norm[0][i][j] = noa;
			norm[1][i][j] = nob;
			norm[2][i][j] = wk[0][i][j];
			norm[3][i][j] = wk[1][i][j];	
		}
	}

    if(lthetax == 0) ltheta = 0;
    else ltheta = lthetay / lthetax;

	printf("lddd: %.3f\n", ltheta);

    DBLFree_3(xd, l, h);
	DBLFree_3(wk, l, h);
    
    return ltheta;
}
double ldd_h2_norm_souwarate_torn(double ***a, double ***b, double ***hpq, double ***norm, double *l3_to, double *l3_no, int *c, int *c_to, double lsigma, int w, int h){
	int i, j, p, q;
	double lthetax = 0;
	double lthetay = 0;
	int count_va = 0;
	double lthetax_torn = 0;
	double lthetay_torn = 0;
	int count_torn = 0;
    double wall = 0;
	double xtu, noa, nob, ltheta, ltheta_torn, ltheta_normal;
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

            noa = xa[0][i][j] * xa[0][i][j] + xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];
			noa = sqrt(noa);
			xtu = xd[1][i][j] * cos(thetak) + xd[2][i][j] * sin(thetak);
			wk[0][i][j] = exp(xa[0][i][j] * xa[0][i][j] / (-2 * lsigma * lsigma));
            wk[1][i][j] = exp(xd[0][i][j] * xd[0][i][j] / (-2 * lsigma * lsigma));

			xd[1][i][j] = xd[1][i][j] - xtu * cos(thetak);  
			xd[2][i][j] = xd[2][i][j] - xtu * sin(thetak);

            nob = xd[0][i][j] * xd[0][i][j] + xd[1][i][j] * xd[1][i][j] + xd[2][i][j] * xd[2][i][j];
			nob = sqrt(nob);

		    lthetax += wk[0][i][j] * noa;
            lthetay += wk[1][i][j] * nob;
			if(noa > 0) count_va++;
            
			if(nob > noa){
		    	lthetax_torn += wk[0][i][j] * noa;
            	lthetay_torn += wk[1][i][j] * nob;		
				count_torn++;		
			}

			norm[0][i][j] = noa;
			norm[1][i][j] = nob;
			norm[2][i][j] = wk[0][i][j];
			norm[3][i][j] = wk[1][i][j];			
		}
	}

    if(lthetax == 0) ltheta = 0;
    else ltheta = lthetay / lthetax;

    if(lthetax_torn == 0){
		ltheta_torn = 0;
		ltheta_normal = ltheta;
	}
    else{
		ltheta_torn = lthetay_torn / lthetax_torn;
		ltheta_normal = (lthetay - lthetay_torn) / (lthetax - lthetax_torn);
	}

	*l3_to = ltheta_torn;
	*l3_no = ltheta_normal;
	*c = count_va;
	*c_to = count_torn;

	//printf("lddd: %.3f, count_va: %d\n", ltheta, count_va);
	//printf("lddd_torn: %.3f, count: %d\n", ltheta_torn, count_torn);
	//printf("lddd_normal: %.3f\n", ltheta_normal);

    DBLFree_3(xd, l, h);
    DBLFree_3(xa, l, h);
	DBLFree_3(wk, l, h);
    
    return ltheta;
}
double l1_h2_norm_souwarate(double ***a, double ***b, double ***hpq, int w, int h){
	int i, j, p, q;
	double lthetax = 0;
	double lthetay = 0;
	double xtu, noa, nob, ltheta;
	double ***xd, ***xa;

	xd = DBLCreate_3(l, w, h);
	xa = DBLCreate_3(l, w, h);

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

            noa = xa[0][i][j] * xa[0][i][j] + xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];
			noa = sqrt(noa);
			xtu = xd[1][i][j] * cos(thetak) + xd[2][i][j] * sin(thetak);

			xd[1][i][j] = xd[1][i][j] - xtu * cos(thetak);  
			xd[2][i][j] = xd[2][i][j] - xtu * sin(thetak);

            nob = xd[0][i][j] * xd[0][i][j] + xd[1][i][j] * xd[1][i][j] + xd[2][i][j] * xd[2][i][j];
			nob = sqrt(nob);

		    lthetax += noa;
            lthetay += nob;		
		}
	}

    if(lthetax == 0) ltheta = 0;
    else ltheta = lthetay / lthetax;

	printf("L1: %.3f\n", ltheta);

    DBLFree_3(xd, l, h);
    DBLFree_3(xa, l, h);

    return ltheta;
}
double l2_h2_relative_error(double ***a, double ***b, double ***hpq, char *iname, int w, int h){
	int i, j, p, q;
	int scount = 0;
	double ltheta = 0;
	double xtu, noa, nob, a_i;
	double ***xd, ***xa;
	char dname[MAX_FILENAME];

	xd = DBLCreate_3(l, w, h);
	xa = DBLCreate_3(l, w, h);

    FILE *fp;

	sprintf(dname, "%s_ldd1_h2_l2_b_a.csv", iname);
    fp = fopen(dname, "a");
	
	if(fp == NULL){
		printf("dame\n");
		return -1;
	}

	fprintf(fp, "no,B_i,A_i\n");




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

        	noa = xa[0][i][j] * xa[0][i][j] + xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];
			noa = sqrt(noa);
			xtu = xd[1][i][j] * cos(thetak) + xd[2][i][j] * sin(thetak);

			xd[1][i][j] = xd[1][i][j] - xtu * cos(thetak);  
			xd[2][i][j] = xd[2][i][j] - xtu * sin(thetak);

            nob = xd[0][i][j] * xd[0][i][j] + xd[1][i][j] * xd[1][i][j] + xd[2][i][j] * xd[2][i][j];
			nob = sqrt(nob);

			if(noa != 0){
				a_i = fabs(nob -noa);
				ltheta += a_i / noa;
				scount++;
				fprintf(fp, "%d,%f,%f\n",scount,noa,a_i);
			}
		}
	}


	fclose(fp);

	ltheta = ltheta / scount;
	printf("L2': %.3f\n", ltheta);

    DBLFree_3(xd, l, h);
    DBLFree_3(xa, l, h);

    return ltheta;
}
double ldd_h2_abnorm_souwarate(double ***a, double ***b, double ***hpq, double ***norm, double lsigma, int w, int h){
	int i, j, p, q;
	double lthetax = 0;
	double lthetay = 0;
    double wall = 0;
	double xtu, noa, nob, ltheta;
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

            noa = xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];
			noa = sqrt(noa);
			xtu = xd[1][i][j] * cos(thetak) + xd[2][i][j] * sin(thetak);
			wk[0][i][j] = exp(xa[0][i][j] * xa[0][i][j] / (-2 * lsigma * lsigma));
            wk[1][i][j] = exp(xd[0][i][j] * xd[0][i][j] / (-2 * lsigma * lsigma));

			xd[1][i][j] = xd[1][i][j] - xtu * cos(thetak);  
			xd[2][i][j] = xd[2][i][j] - xtu * sin(thetak);

            nob = xd[1][i][j] * xd[1][i][j] + xd[2][i][j] * xd[2][i][j];
			nob = sqrt(nob);

		    lthetax += wk[0][i][j] * noa;
            lthetay += wk[1][i][j] * nob;

			norm[0][i][j] = noa;
			norm[1][i][j] = nob;
			norm[2][i][j] = wk[0][i][j];
			norm[3][i][j] = wk[1][i][j];	
		}
	}

    if(lthetax == 0) ltheta = 0;
    else ltheta = lthetay / lthetax;

	printf("lddd: %.3f\n", ltheta);

    DBLFree_3(xd, l, h);
	DBLFree_3(wk, l, h);
    
    return ltheta;
}
void projection0(double ***a, double ***b, int w, int h){
    int i, j;
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
		}
	}
}
void projection_c(double ***a, double ***b, int w, int h){
    int i, j;
	double rotate, atu, cborder2, chroma, hangle;

    for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			rotate = thetak + PI;

			atu = a[1][i][j] * cos(rotate) + a[2][i][j] * sin(rotate);

			b[0][i][j] = a[0][i][j];
			b[1][i][j] = a[1][i][j] - atu * cos(rotate);
			b[2][i][j] = a[2][i][j] - atu * sin(rotate);
		}
	}
}
void escore(PPMStruct *img_ref_p, PPMStruct *img_ref_f, double ***a, double ***b, double ***hpq, double *ccpr, double *ccfr, double *escore, int w, int h){
    int	i, j, p, q;
	double mother_p = 0;
	double child_p = 0;
	double mother_f = 0;
	double child_f = 0;
	double noa, nob, np, nf, ne;
	double ***b2, ***xb, ***xa;

	b2 = DBLCreate_3(l, w, h);
	xb = DBLCreate_3(l, w, h);
	xa = DBLCreate_3(l, w, h);

	projection0(b, b2, w, h);

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			p = hpq[0][i][j];
			q = hpq[1][i][j];

			xb[0][i][j] = b2[0][i][j] - b2[0][p][q];
			xb[1][i][j] = b2[1][i][j] - b2[1][p][q];  
			xb[2][i][j] = b2[2][i][j] - b2[2][p][q]; 

            nob = xb[0][i][j] * xb[0][i][j] + xb[1][i][j] * xb[1][i][j] + xb[2][i][j] * xb[2][i][j];
			nob = sqrt(nob);

            xa[0][i][j] = a[0][i][j] - a[0][p][q];
			xa[1][i][j] = a[1][i][j] - a[1][p][q];  
			xa[2][i][j] = a[2][i][j] - a[2][p][q];  

            noa = xa[0][i][j] * xa[0][i][j] + xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];
			noa = sqrt(noa);

			//if(i==127 && j ==97) printf("2色覚でのΔL*：%f、Δa*：%f、Δb*：%f\n", xb[0][i][j], xb[1][i][j], xb[2][i][j]);
			//if(i==127 && j ==97) printf("3色覚での色差：%f、2色覚での色差：%f\n", noa, nob);

            if(noa > ptau3siki){
                mother_p++;
                img_ref_p->p[0][i][j] = 0;
			    img_ref_p->p[1][i][j] = 255;
			    img_ref_p->p[2][i][j] = 0;
                if(nob > ptau2siki){
                    child_p++;
					img_ref_p->p[0][i][j] = 255;
			        img_ref_p->p[1][i][j] = 0;
			        img_ref_p->p[2][i][j] = 0;
                }
            }
			else{
                img_ref_p->p[0][i][j] = 0;
			    img_ref_p->p[1][i][j] = 0;
			    img_ref_p->p[2][i][j] = 0; 
            }

			if(nob > ftau2siki){
                mother_f++;
                img_ref_f->p[0][i][j] = 0;
			    img_ref_f->p[1][i][j] = 255;
			    img_ref_f->p[2][i][j] = 0;
                if(noa < ftau3siki){
                    child_f++;
                    img_ref_f->p[0][i][j] = 255;
			        img_ref_f->p[1][i][j] = 0;
			        img_ref_f->p[2][i][j] = 0;
                }
            }
            else{
                img_ref_f->p[0][i][j] = 0;
			    img_ref_f->p[1][i][j] = 0;
			    img_ref_f->p[2][i][j] = 0; 
            }

		}
	}

    if(mother_p == 0) np = 0;
    else np = child_p / mother_p;

    if(mother_f == 0) nf = 0;
    else nf = 1 - child_f / mother_f;

	ne = 2 * np * nf / (np + nf);

	*ccpr = np;
	*ccfr = nf; 
	*escore = ne; 

	//printf("ccpr: %.3f\n", np);
	//printf("ccfr: %.3f\n", nf);
	//printf("escore: %.3f\n", ne);

    DBLFree_3(b2, l, h);
    DBLFree_3(xb, l, h);
    DBLFree_3(xa, l, h);
}
void escore_is(PPMStruct *img_ref_p, PPMStruct *img_ref_f, double ***a, double ***b, double ***hpq, double *ccpr, double *ccfr, double *escore, int w, int h){
    int	i, j, p, q;
	double mother_p = 0;
	double child_p = 0;
	double mother_f = 0;
	double child_f = 0;
	double noa, nob, np, nf, ne;
	double ***b2, ***xb, ***xa;

	b2 = DBLCreate_3(l, w, h);
	xb = DBLCreate_3(l, w, h);
	xa = DBLCreate_3(l, w, h);

	projection_c(b, b2, w, h);

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			p = hpq[0][i][j];
			q = hpq[1][i][j];

			xb[0][i][j] = b2[0][i][j] - b2[0][p][q];
			xb[1][i][j] = b2[1][i][j] - b2[1][p][q];  
			xb[2][i][j] = b2[2][i][j] - b2[2][p][q]; 

            nob = xb[0][i][j] * xb[0][i][j] + xb[1][i][j] * xb[1][i][j] + xb[2][i][j] * xb[2][i][j];
			nob = sqrt(nob);

            xa[0][i][j] = a[0][i][j] - a[0][p][q];
			xa[1][i][j] = a[1][i][j] - a[1][p][q];  
			xa[2][i][j] = a[2][i][j] - a[2][p][q];  

            noa = xa[0][i][j] * xa[0][i][j] + xa[1][i][j] * xa[1][i][j] + xa[2][i][j] * xa[2][i][j];
			noa = sqrt(noa);

			//if(i==127 && j ==97) printf("2色覚でのΔL*：%f、Δa*：%f、Δb*：%f\n", xb[0][i][j], xb[1][i][j], xb[2][i][j]);
			//if(i==127 && j ==97) printf("3色覚での色差：%f、2色覚での色差：%f\n", noa, nob);

            if(noa > ptau3siki){
                mother_p++;
                img_ref_p->p[0][i][j] = 0;
			    img_ref_p->p[1][i][j] = 255;
			    img_ref_p->p[2][i][j] = 0;
                if(nob > ptau2siki){
                    child_p++;
					img_ref_p->p[0][i][j] = 255;
			        img_ref_p->p[1][i][j] = 0;
			        img_ref_p->p[2][i][j] = 0;
                }
            }
			else{
                img_ref_p->p[0][i][j] = 0;
			    img_ref_p->p[1][i][j] = 0;
			    img_ref_p->p[2][i][j] = 0; 
            }

			if(nob > ftau2siki){
                mother_f++;
                img_ref_f->p[0][i][j] = 0;
			    img_ref_f->p[1][i][j] = 255;
			    img_ref_f->p[2][i][j] = 0;
                if(noa < ftau3siki){
                    child_f++;
                    img_ref_f->p[0][i][j] = 255;
			        img_ref_f->p[1][i][j] = 0;
			        img_ref_f->p[2][i][j] = 0;
                }
            }
            else{
                img_ref_f->p[0][i][j] = 0;
			    img_ref_f->p[1][i][j] = 0;
			    img_ref_f->p[2][i][j] = 0; 
            }

		}
	}

    if(mother_p == 0) np = 0;
    else np = child_p / mother_p;

    if(mother_f == 0) nf = 0;
    else nf = 1 - child_f / mother_f;

	ne = 2 * np * nf / (np + nf);

	*ccpr = np;
	*ccfr = nf; 
	*escore = ne; 

	//printf("ccpr: %.3f\n", np);
	//printf("ccfr: %.3f\n", nf);
	//printf("escore: %.3f\n", ne);

    DBLFree_3(b2, l, h);
    DBLFree_3(xb, l, h);
    DBLFree_3(xa, l, h);
}
void w2_xnorm(PPMStruct *img_ref_w2, double ***a, int w, int h){
    int i, j, rr, gg, bb;
	double light;

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			light = a[0][i][j] * a[2][i][j];

		    if(light>100) light = 100;
            lab2rgb(&rr, &gg, &bb, light, 0, 0);

			img_ref_w2->p[0][i][j] = rr;
			img_ref_w2->p[1][i][j] = gg;
			img_ref_w2->p[2][i][j] = bb;
		}
	}
}
double ph_v3(double x, double y){
	double ph;

	ph = x - y;		
	if(ph < 0){
        ph = 0;
    }

	return ph;
}
double ph_v4(double x, double y){
	double ph;

	ph = perm * x - y;		
	if(ph < 0){
        ph = 0;
    }

	return ph;
}
void v_a(double ***a, double ***b, double ***hpq, double *v_1, double *v_2, double *v_3, double *v_4, double esigma, char *iname, int w, int h){
	int i, j, p, q;
	double u_out = 0;
	double u_in1 = 0;
	double u_in2 = 0;
	double u_out3 = 0;   	
    double u_in3 = 0;
	double u_out4 = 0;   	
    double u_in4 = 0;
	double yi_out, yi_in, xi_in, v1, v2, v3, v4;
	double ***b2, ***a2, ***y_out, ***y_in, ***x_in, ***wk;
	char fname[MAX_FILENAME];

	b2 = DBLCreate_3(l, w, h);
	a2 = DBLCreate_3(l, w, h);
	y_out = DBLCreate_3(l, w, h);
	y_in = DBLCreate_3(l, w, h);
	x_in = DBLCreate_3(l, w, h);
	wk = DBLCreate_3(l, w, h);

	projection0(b, b2, w, h);
	projection0(a, a2, w, h);

    puts("a");

    FILE *fp;
	sprintf(fname, "%s_ldd1_a.csv", iname);
    fp = fopen(fname, "a");
	

	fprintf(fp, "i,j,x_in,y_in,y_out,w2\n");

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			p = hpq[0][i][j];
			q = hpq[1][i][j];

			y_out[0][i][j] = b2[0][i][j] - b2[0][p][q];
			y_out[1][i][j] = b2[1][i][j] - b2[1][p][q];  
			y_out[2][i][j] = b2[2][i][j] - b2[2][p][q]; 

			y_in[0][i][j] = a2[0][i][j] - a2[0][p][q];
			y_in[1][i][j] = a2[1][i][j] - a2[1][p][q];  
			y_in[2][i][j] = a2[2][i][j] - a2[2][p][q]; 

            x_in[0][i][j] = a[0][i][j] - a[0][p][q];
			x_in[1][i][j] = a[1][i][j] - a[1][p][q];  
			x_in[2][i][j] = a[2][i][j] - a[2][p][q];

            yi_out = y_out[0][i][j] * y_out[0][i][j] + y_out[1][i][j] * y_out[1][i][j] + y_out[2][i][j] * y_out[2][i][j];
			yi_out = sqrt(yi_out);

            yi_in = y_in[0][i][j] * y_in[0][i][j] + y_in[1][i][j] * y_in[1][i][j] + y_in[2][i][j] * y_in[2][i][j];
			yi_in = sqrt(yi_in);

            xi_in = x_in[0][i][j] * x_in[0][i][j] + x_in[1][i][j] * x_in[1][i][j] + x_in[2][i][j] * x_in[2][i][j];
			xi_in = sqrt(xi_in);		

            wk[0][i][j] = exp(x_in[0][i][j] * x_in[0][i][j] / (-2 * esigma * esigma));

            //if(fabs(yi_out - yi_in) > 0.000001) printf("xi_in:%f, yi_out:%f, yi_in:%f\n", xi_in, yi_out, yi_in);

			u_out += wk[0][i][j] * fabs(yi_out - xi_in);
			u_in1 += wk[0][i][j] * xi_in;
			u_in2 += wk[0][i][j] * fabs(yi_in - xi_in);

            u_out3 += wk[0][i][j] * ph_v3(xi_in, yi_out);
            u_in3  += wk[0][i][j] * ph_v3(xi_in, yi_in); 
            u_out4 += wk[0][i][j] * ph_v4(xi_in, yi_out);
            u_in4  += wk[0][i][j] * ph_v4(xi_in, yi_in);   

            fprintf(fp, "%d,%d,%f,%f,%f,%f\n",i,j,xi_in,yi_in,yi_out,wk[0][i][j]);         
		}
	}
 
	fclose(fp);	

	sprintf(fname, "%s_ldd1_V4_a.csv", iname);
    fp = fopen(fname, "a");
	
	fprintf(fp, "iname,perm,u_out4,u_in4\n");
	fprintf(fp, "%s,%f,%f,%f\n",iname,perm,u_out4,u_in4); 
	fclose(fp);	


	v1 = u_out / u_in1;
	*v_1 = v1;
	v2 = u_out / u_in2;
	*v_2 = v2;
	v3 = u_out3 / u_in3;
	*v_3 = v3;
	v4 = u_out4 / u_in4;
	*v_4 = v4;

    DBLFree_3(b2, l, h);
    DBLFree_3(a2, l, h);
    DBLFree_3(y_out, l, h);
    DBLFree_3(y_in, l, h);
    DBLFree_3(x_in, l, h);
    DBLFree_3(wk, l, h);
}
void v_b(double ***a, double ***b, double ***hpq, double *v_1, double *v_2, double *v_3, double *v_4, double esigma, char *iname, int w, int h){
	int i, j, p, q;
	double u_out = 0;
	double u_in1 = 0;
	double u_in2 = 0;
	double u_out3 = 0;   	
    double u_in3 = 0;
	double u_out4 = 0;   	
    double u_in4 = 0;
	double yi_out, yi_in, xi_in, v1, v2, v3, v4;
	double ***b2, ***a2, ***y_out, ***y_in, ***x_in, ***wk;
	char fname[MAX_FILENAME];

	b2 = DBLCreate_3(l, w, h);
	a2 = DBLCreate_3(l, w, h);
	y_out = DBLCreate_3(l, w, h);
	y_in = DBLCreate_3(l, w, h);
	x_in = DBLCreate_3(l, w, h);
	wk = DBLCreate_3(l, w, h);

	projection0(b, b2, w, h);
	projection0(a, a2, w, h);

    puts("a");

    FILE *fp;
	sprintf(fname, "%s_ldd1_b.csv", iname);
    fp = fopen(fname, "a");
	

	fprintf(fp, "i,j,x_in,y_in,y_out,w2\n");

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			p = hpq[0][i][j];
			q = hpq[1][i][j];

			y_out[0][i][j] = b2[0][i][j] - b2[0][p][q];
			y_out[1][i][j] = b2[1][i][j] - b2[1][p][q];  
			y_out[2][i][j] = b2[2][i][j] - b2[2][p][q]; 

			y_in[0][i][j] = a2[0][i][j] - a2[0][p][q];
			y_in[1][i][j] = a2[1][i][j] - a2[1][p][q];  
			y_in[2][i][j] = a2[2][i][j] - a2[2][p][q]; 

            x_in[0][i][j] = a[0][i][j] - a[0][p][q];
			x_in[1][i][j] = a[1][i][j] - a[1][p][q];  
			x_in[2][i][j] = a[2][i][j] - a[2][p][q];

            yi_out = y_out[0][i][j] * y_out[0][i][j] + y_out[1][i][j] * y_out[1][i][j] + y_out[2][i][j] * y_out[2][i][j];
			yi_out = sqrt(yi_out);

            yi_in = y_in[0][i][j] * y_in[0][i][j] + y_in[1][i][j] * y_in[1][i][j] + y_in[2][i][j] * y_in[2][i][j];
			yi_in = sqrt(yi_in);

            xi_in = x_in[0][i][j] * x_in[0][i][j] + x_in[1][i][j] * x_in[1][i][j] + x_in[2][i][j] * x_in[2][i][j];
			xi_in = sqrt(xi_in);		

            wk[0][i][j] = exp(x_in[0][i][j] * x_in[0][i][j] / (-2 * esigma * esigma));

            //if(fabs(yi_out - yi_in) > 0.000001) printf("xi_in:%f, yi_out:%f, yi_in:%f\n", xi_in, yi_out, yi_in);

			u_out += wk[0][i][j] * fabs(yi_out - xi_in);
			u_in1 += wk[0][i][j] * xi_in;
			u_in2 += wk[0][i][j] * fabs(yi_in - xi_in);

            u_out3 += wk[0][i][j] * ph_v3(xi_in, yi_out);
            u_in3  += wk[0][i][j] * ph_v3(xi_in, yi_in); 
            u_out4 += wk[0][i][j] * ph_v4(xi_in, yi_out);
            u_in4  += wk[0][i][j] * ph_v4(xi_in, yi_in);   

            fprintf(fp, "%d,%d,%f,%f,%f,%f\n",i,j,xi_in,yi_in,yi_out,wk[0][i][j]);         
		}
	}
 
	fclose(fp);	

	sprintf(fname, "%s_ldd1_V4_b.csv", iname);
    fp = fopen(fname, "a");
	
	fprintf(fp, "iname,perm,u_out4,u_in4\n");
	fprintf(fp, "%s,%f,%f,%f\n",iname,perm,u_out4,u_in4); 
	fclose(fp);	


	v1 = u_out / u_in1;
	*v_1 = v1;
	v2 = u_out / u_in2;
	*v_2 = v2;
	v3 = u_out3 / u_in3;
	*v_3 = v3;
	v4 = u_out4 / u_in4;
	*v_4 = v4;

    DBLFree_3(b2, l, h);
    DBLFree_3(a2, l, h);
    DBLFree_3(y_out, l, h);
    DBLFree_3(y_in, l, h);
    DBLFree_3(x_in, l, h);
    DBLFree_3(wk, l, h);
}
void v_c(double ***a, double ***b, double ***hpq, double *v_1, double *v_2, double *v_3, double *v_4, double esigma, char *iname, int w, int h){
	int i, j, p, q;
	double u_out = 0;
	double u_in1 = 0;
	double u_in2 = 0;
	double u_out3 = 0;   	
    double u_in3 = 0;
	double u_out4 = 0;   	
    double u_in4 = 0;
	double yi_out, yi_in, xi_in, v1, v2, v3, v4;
	double ***b2, ***a2, ***y_out, ***y_in, ***x_in, ***wk;
	char fname[MAX_FILENAME];

	b2 = DBLCreate_3(l, w, h);
	a2 = DBLCreate_3(l, w, h);
	y_out = DBLCreate_3(l, w, h);
	y_in = DBLCreate_3(l, w, h);
	x_in = DBLCreate_3(l, w, h);
	wk = DBLCreate_3(l, w, h);

	projection_c(b, b2, w, h);
	projection_c(a, a2, w, h);

    puts("a");

    FILE *fp;
	sprintf(fname, "%s_ldd1_c.csv", iname);
    fp = fopen(fname, "a");
	

	fprintf(fp, "i,j,x_in,y_in,y_out,w2\n");

	for(i = 0; i < h; i++){
       	for(j = 0; j < w; j++){
			p = hpq[0][i][j];
			q = hpq[1][i][j];

			y_out[0][i][j] = b2[0][i][j] - b2[0][p][q];
			y_out[1][i][j] = b2[1][i][j] - b2[1][p][q];  
			y_out[2][i][j] = b2[2][i][j] - b2[2][p][q]; 

			y_in[0][i][j] = a2[0][i][j] - a2[0][p][q];
			y_in[1][i][j] = a2[1][i][j] - a2[1][p][q];  
			y_in[2][i][j] = a2[2][i][j] - a2[2][p][q]; 

            x_in[0][i][j] = a[0][i][j] - a[0][p][q];
			x_in[1][i][j] = a[1][i][j] - a[1][p][q];  
			x_in[2][i][j] = a[2][i][j] - a[2][p][q];

            yi_out = y_out[0][i][j] * y_out[0][i][j] + y_out[1][i][j] * y_out[1][i][j] + y_out[2][i][j] * y_out[2][i][j];
			yi_out = sqrt(yi_out);

            yi_in = y_in[0][i][j] * y_in[0][i][j] + y_in[1][i][j] * y_in[1][i][j] + y_in[2][i][j] * y_in[2][i][j];
			yi_in = sqrt(yi_in);

            xi_in = x_in[0][i][j] * x_in[0][i][j] + x_in[1][i][j] * x_in[1][i][j] + x_in[2][i][j] * x_in[2][i][j];
			xi_in = sqrt(xi_in);		

            wk[0][i][j] = exp(x_in[0][i][j] * x_in[0][i][j] / (-2 * esigma * esigma));

            //if(fabs(yi_out - yi_in) > 0.000001) printf("xi_in:%f, yi_out:%f, yi_in:%f\n", xi_in, yi_out, yi_in);

			u_out += wk[0][i][j] * fabs(yi_out - xi_in);
			u_in1 += wk[0][i][j] * xi_in;
			u_in2 += wk[0][i][j] * fabs(yi_in - xi_in);

            u_out3 += wk[0][i][j] * ph_v3(xi_in, yi_out);
            u_in3  += wk[0][i][j] * ph_v3(xi_in, yi_in);
            u_out4 += wk[0][i][j] * ph_v4(xi_in, yi_out);
            u_in4  += wk[0][i][j] * ph_v4(xi_in, yi_in);

            fprintf(fp, "%d,%d,%f,%f,%f,%f\n",i,j,xi_in,yi_in,yi_out,wk[0][i][j]);
		}
	}
 
	fclose(fp);	

	sprintf(fname, "%s_ldd1_V4_c.csv", iname);
    fp = fopen(fname, "a");
	
	fprintf(fp, "iname,perm,u_out4,u_in4\n");
	fprintf(fp, "%s,%f,%f,%f\n",iname,perm,u_out4,u_in4); 
	fclose(fp);	

	v1 = u_out / u_in1;
	*v_1 = v1;
	v2 = u_out / u_in2;
	*v_2 = v2;
	v3 = u_out3 / u_in3;
	*v_3 = v3;
	v4 = u_out4 / u_in4;
	*v_4 = v4;

    DBLFree_3(b2, l, h);
    DBLFree_3(a2, l, h);
    DBLFree_3(y_out, l, h);
    DBLFree_3(y_in, l, h);
    DBLFree_3(x_in, l, h);
    DBLFree_3(wk, l, h);
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

