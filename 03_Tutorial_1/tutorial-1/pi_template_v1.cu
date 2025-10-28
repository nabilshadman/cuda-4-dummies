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
 *               n.b. this is a CUDA implementation where the core
 *                    part of the calculation --- computing distances 
 *                    from the origin --- is carried out on the GPU
 *                    using CUDA-managed unified memory;
 *               n.b.2 here we simply want to increase N, the number
 *                     of random points considered, and examine the
 *                     resulting level of accuracy;
 *               n.b.3 however, since the number of threads per 
 *                     threadblock is limited to approximately 1024,
 *                     we now need to work with a block grid of 
 *                     multiple threadblocks; 
 * compilation:  nvcc ./pi_solution_v1.cu 
 * usage:        ./a.out  
 * result:       computed pi:     3.080000000  (N 100)
 *               computed pi:     ?            (N 1000)
 *               computed pi:     ?            (N 10000)
 *               computed pi:     ?            (N 100000)
 *               machine  pi:     3.141592654
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 1000
#define NTB 256 


// review the following kernel and extend it in case we
// needed to consider multiple threadblocks
__global__ void distances(double *x, double *y, double *r)
{
    int i;

    i = threadIdx.x;
    r[i] = sqrt( (x[i] * x[i]) + (y[i] * y[i]) );

    return;
}



int main()
{
  int i, n, nmb_points_outside_unit_circle, nmb_points_inside_unit_circle;
  double *x, *y, *r, ratio_partial_areas, computed_pi;

 /*
  * intialize random number generator 
  */
  srand((unsigned) time(NULL));

 // reconsider and adjust the following section and take into 
 // account that array dimensions, N,  need not necessarily be 
 // an integral multiple of the size of the threadblock, NTB
 /*
  * allocate memory for all arrays making use of
  * CUDA-managed unified memory
  */
  cudaMallocManaged(&x, N * sizeof(double));
  cudaMallocManaged(&y, N * sizeof(double));
  cudaMallocManaged(&r, N * sizeof(double));

 /*
  * generate random coordinates in the first quadrant 
  * i.e. with x,y-values in the range [0,1]
  */
  for (i = 0; i < N; i++) {
      x[i] = (double) rand() / (double) RAND_MAX;
      y[i] = (double) rand() / (double) RAND_MAX;
      // printf("%10d%16.9lf%16.9lf\n", i, x[i], y[i]);
  }

 // reconsider the following kernel launch and extend it in 
 // case we needed to account for multiple threadblocks
 /*
  * launch a GPU kernel for the computation of distances
  */
  distances<<<1, N>>>(&x[0], &y[0], &r[0]);
  cudaDeviceSynchronize();

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
  cudaFree(r);
  cudaFree(y);
  cudaFree(x);


  return 0;
}
