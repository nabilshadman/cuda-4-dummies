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
 *               n.b.4 in this version we also want to carry out
 *                     partial sums, s[], by the threadblocks, i.e. 
 *                     instead of having the host code finally run over 
 *                     all elements in the r[] array, do already provide
 *                     partial results s[] (counts of points outside
 *                     the unit circle) by individual threadblocks; 
 *                     
 * compilation:  nvcc ./pi_solution_v3.cu 
 * usage:        /usr/bin/time -v ./a.out  
 *               nsys nvprof ./a.out  
 * result:       computed pi:     3.146253680
 *               machine  pi:     3.141592654
 *               timings: 0.74user 5.58system 0:06.46elapsed 97%CPU 
 *                        2bcompared2
 *                        1.13user 6.24system 0:07.51elapsed 98%CPU based on ./pi_solution_v2.cu
 *               kernel exe time: ~0.7006 s  
 *                                2bcompared2 
 *                                ~0.62104 s based on ./pi_solution_v2.cu
 *               
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <curand_kernel.h>

#define N 500000000
#define NTB 256


__global__ void distances3(double *x, double *y, double *r, int *s)
{
    int i;
    curandState state;

   /*
    * have thread 0 act as a kind of master thread and use
    * its corresponding element in s[] to accumulate all r[]
    * results of the threadblock
    */
    if (threadIdx.x == 0) {
       s[blockIdx.x] = 0;
    }
    __syncthreads();

    i = (blockIdx.x * blockDim.x) + threadIdx.x;

    curand_init((unsigned long long)clock() + i, 0, 0, &state);
    x[i] = curand_uniform_double(&state);
    y[i] = curand_uniform_double(&state);
    r[i] = sqrt( (x[i] * x[i]) + (y[i] * y[i]) );

   /*
    * since all threads in the threadblock want to update s[]  
    * at roughly the same point in time we need to be careful
    * with race conditions, hence go for an atomic operation;
    * in addition, the final threadblock may be incomplete, so
    * we should take care not to include points that have not
    * been requested, i.e. with indices beyond N
    */
    if (i < N) {
       atomicAdd(&s[blockIdx.x], (int) r[i]);
    }
    

    return;
}



int main()
{
  int i, n, *s, nmb_points_outside_unit_circle, nmb_points_inside_unit_circle;
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
  cudaMallocManaged(&s, n * sizeof(int));
  printf("N=%-12dNTB=%-12dn=%-12d\n", N, NTB, n);


 /*
  * launch a GPU kernel for the computation of distances
  * n.b. now the kernel execution configuration is a true
  *      2-level one, consisting of a bunch of threadblocks
  *      forming the blockgrid;
  * n.b.2 now the kernel code also does the random number
  *       assignment of x,y coordinates 
  * n.b.3 in addition partial results, s[], are already 
  *       computed at the level of threadblocks so that 
  *       the host code simply needs to quickly sum up
  *       the reduced list of s[] in order to get all
  *       points counted that are outside the unit circle
  */
  distances3<<<n, NTB>>>(&x[0], &y[0], &r[0], &s[0]);
  cudaDeviceSynchronize(); 

 /*
  * count all cases of r[i] > 1.0 because these are points 
  * outside the unit circle; consequently, N minus that number
  * will be the number of points inside the unit circle;
  * n.b. since the major part of this task has already been
  *      carried out by individual threadblocks, here we
  *      simply need to sum up the partial results
  */
  nmb_points_outside_unit_circle = 0;
  for (i = 0; i < n; i++) {
      nmb_points_outside_unit_circle += s[i];
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
  cudaFree(s);
  cudaFree(r);
  cudaFree(y);
  cudaFree(x);


  return 0;
}
