/* 
 * purpose:      calculate the full Coulomb interaction for all 
 *               atoms found in the structure file MOLPDB that has 
 *               been extended with partial charges and vdW parameters; 
 *               this version may just serve for providing a reference;
 * compilation:  gcc ./coulomb_v0.c -lm  
 * usage:        ./a.out  
 * result:       coulomb sum: -230.611048407
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4383


int main()
{
  int i, j;
  double *x, *y, *z, *q, cs, tx, ty, tz, d;
  FILE *fp1;


 /*
  * allocate memory for all arrays used
  */
  x = (double *) malloc(N * sizeof(double));
  y = (double *) malloc(N * sizeof(double));
  z = (double *) malloc(N * sizeof(double));
  q = (double *) malloc(N * sizeof(double));

 /*
  * reading the MOLPDB file, respectively coordinates and 
  * partial charges
  */
  fp1 = fopen("MOLPDB", "r");
  for (i = 0; i < N; i++) {
      fscanf(fp1, "%*s%*d%*s%*s%*d%lf%lf%lf%lf%*lf%*lf%*s\n", &x[i], &y[i], &z[i], &q[i]);
      // printf("%6d%8.3lf%8.3lf%8.3lf%10.5lf\n", i, x[i], y[i], z[i], q[i]);
  }
  fclose(fp1);

 /*
  * compute the Coulomb interaction from a full double sum
  * over all pairs, just neglecting the self-interaction
  */
  cs = (double) 0;
  for (i = 0; i < N-1; i++) {
      for (j = i+1; j < N; j++) {
          tx = x[i] - x[j];
          ty = y[i] - y[j];
          tz = z[i] - z[j];
          d = sqrt( (tx * tx) + (ty * ty) + (tz * tz) );
          cs += q[i] * q[j] / d;
      }
  }
  printf("coulomb sum: %14.9lf\n", cs);


 /*
  * and not to forget, free the allocated memory 
  */
  free(q);
  free(z);
  free(y);
  free(x);


  return 0;
}
