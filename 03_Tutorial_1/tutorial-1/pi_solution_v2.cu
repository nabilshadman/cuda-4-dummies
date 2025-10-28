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
 *               n.b.2 N shall be set to a value of 5.0x10^8;
 *               n.b.3 in this version we also want to include 
 *                     random coordinate generation in the kernel code 
 *                     and compare runtimes to the previous version 
 *                     where random numbers were calculated exclusively  
 *                     on the host
 * compilation:  nvcc ./pi_solution_v2.cu 
 * usage:        nsys nvprof ./a.out  
 * result:       kernel exe time: ~1.372 s  
 *               2bcompared2 
 *               kernel exe time: ~2.139 s based on ./pi_solution_v1.cu
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>

#define N 500000000
#define NTB 256 


__global__ void distances2(double *x, double *y, double *r)
{
    int i;
    curandState state;

    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    curand_init((unsigned long long)clock() + i, 0, 0, &state);
    x[i] = curand_uniform_double(&state);
    y[i] = curand_uniform_double(&state);
    r[i] = sqrt( (x[i] * x[i]) + (y[i] * y[i]) );

    return;
}



int main()
{
  int i, n, nmb_points_outside_unit_circle, nmb_points_inside_unit_circle;
  double *x, *y, *r, ratio_partial_areas, computed_pi;


 /*
  * allocate memory for all arrays making use of
  * CUDA-managed unified memory
  * n.b. if N is not an integral multiple of NTB, the number of 
  *      threads in the threadblock, we shall do some padding of 
  *      arrays to make our lives easier for the upcoming kernel 
  *      launches, so that every thread involved will really have
  *      a corresponding array element to work on regardless if
  *      the latter will (or will not) be considered in the final
  *      computation of pi;
  */
  if (N % NTB != 0) {
     n = (N / NTB) + 1;
  } 
  else {
     n = N / NTB;
  }
  cudaMallocManaged(&x, n * NTB * sizeof(double));
  cudaMallocManaged(&y, n * NTB * sizeof(double));
  cudaMallocManaged(&r, n * NTB * sizeof(double));
  printf("N=%-12dNTB=%-12dn=%-12d\n", N, NTB, n);


 /*
  * launch a GPU kernel for the computation of distances
  * n.b. now the kernel execution configuration is a true
  *      2-level one, consisting of a bunch of threadblocks
  *      forming the blockgrid;
  * n.b.2 now the kernel code also does the random number
  *       assignment of x,y coordinates 
  */
  distances2<<<n, NTB>>>(&x[0], &y[0], &r[0]);
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
