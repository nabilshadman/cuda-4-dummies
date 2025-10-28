/* 
 * purpose:      calculate the full Coulomb interaction for all 
 *               atoms found in the structure file MOLPDB that has 
 *               been extended with partial charges and vdW parameters; 
 * compilation:  nvcc ./coulomb_template_v0.cu
 * usage:        ./a.out  
 * result:       coulomb sum: ?
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4383


// include some kernel code here, basically fixing the Coulomb potentials  
// i.e. individual actions on a particular atom i caused by all other atoms
//      with j > i


int main()
{
  int i, j;
  double *x, *y, *z, *q, cs, tx, ty, tz, d;
  FILE *fp1;


 /*
  * allocate memory for all arrays making use of
  * CUDA-managed unified memory
  * n.b. if N is not an integral multiple of NTB, the number of 
  *      threads in the threadblock, we shall do some padding of 
  *      arrays to make our lives easier for the upcoming kernel 
  *      launches, so that every thread involved will really have
  *      a corresponding array element to work on; padding here 
  *      means coming up with dummy particles of partial charge 0,
  *      and some non-interferring dummy coordinates so that their
  *      contribution to the Coulomb potential will be exactly 0;
  */
  if (N % NTB != 0) {
     n = (N / NTB) + 1;
  } 
  else {
     n = N / NTB;
  }

  // change memory allocation to be of type CUDA-managed unified memory
  // also introduce an additional array for partial results
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


  // substitute the following section with an appropriate kernel call
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


  // change freeing of memory to the corresponding CUDA type
  free(q);
  free(z);
  free(y);
  free(x);


  return 0;
}
