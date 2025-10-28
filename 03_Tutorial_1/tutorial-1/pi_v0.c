/* 
 * purpose:      calculate pi from the ratio of partial areas of
 *               the unit circle and the unit square; so essentially 
 *               we define a set of random x,y coordinates and count
 *               those points that fall inside the unit circle and
 *               take this number as representative of the partial
 *               area of the unit circle while all points altogether
 *               will naturally be members of the partial area of the 
 *               unit square. Since we exclusively consider the first
 *               quadrant only, the aforementioned ratio is
 *                3.14159/4 : 1   hence we can simply approximate
 *               the value of pi from 
 *               4 * #points_inside_unit_circle / #points_considered
 * compilation:  gcc ./pi_v0.c -lm
 * usage:        ./a.out  
 * result:       computed pi:     3.400000000  (and variations thereof)
 *               machine  pi:     3.141592654
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 100


int main()
{
  int i, nmb_points_outside_unit_circle, nmb_points_inside_unit_circle;
  double *x, *y, *r, ratio_partial_areas, computed_pi;

 /*
  * intialize random number generator 
  */
  srand((unsigned) time(NULL));

 /*
  * allocate memory for all arrays used
  */
  x = (double *) malloc(N * sizeof(double));
  y = (double *) malloc(N * sizeof(double));
  r = (double *) malloc(N * sizeof(double));

 /*
  * generate random coordinates in the first quadrant 
  * i.e. with x,y-values in the range [0,1]
  */
  for (i = 0; i < N; i++) {
      x[i] = (double) rand() / (double) RAND_MAX;
      y[i] = (double) rand() / (double) RAND_MAX;
      // printf("%10d%16.9lf%16.9lf\n", i, x[i], y[i]);
  }

 /*
  * compute distance from origin for all random points 
  */
  for (i = 0; i < N; i++) {
      r[i] = sqrt( (x[i] * x[i]) + (y[i] * y[i]) );
  }

 /*
  * count all cases of r[i] > 1.0 because these are points 
  * outside the unit circle; consequently, N minus that number
  * will be the number of points inside the unit circle;
  */
  nmb_points_outside_unit_circle = 0;
  for (i = 0; i < N; i++) {
      nmb_points_outside_unit_circle += (int) r[i];
      // printf("%10d%16.9lf%10d\n", i, r[i], (int) r[i]);
  }
  nmb_points_inside_unit_circle = N - nmb_points_outside_unit_circle;

 /*
  * for well-distributed random points the fraction of points 
  * falling inside the unit circle should reflect the ratio of 
  * partial areas, i.e.  pi/4 : 1  hence we can approximate the 
  * value of pi from this  
  */
  ratio_partial_areas = (double) nmb_points_inside_unit_circle / (double) N;
  computed_pi = 4.0e+00 * ratio_partial_areas;
  printf("computed pi:%16.9lf\n", computed_pi);
  printf("machine  pi:%16.9lf\n", M_PI);

 /*
  * and not to forget, free the allocated memory 
  */
  free(r);
  free(y);
  free(x);


  return 0;
}
