/* 
 * purpose:      calculate the full Coulomb interaction for all 
 *               atoms found in the structure file MOLPDB that has 
 *               been extended with partial charges and vdW parameters; 
 *               n.b. this is a CUDA implementation where individual
 *                    partial sums corresponding to a particular atom 
 *                    will be carried out on the GPU using 
 *                    CUDA-managed unified memory;
 * compilation:  nvcc ./coulomb_solution_v0.cu
 * usage:        ./a.out  
 * result:       coulomb sum: -230.611048407
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4383
#define NTB 256 


__global__ void coulomb(int NN, double *x, double *y, double *z, double *q, double *cp)
{
    int i, j;
    double pcs, rx, ry, rz, r;

    i = (blockIdx.x * blockDim.x) + threadIdx.x;
    pcs = (double) 0;
    
    for (j = i + 1; j < NN; j++) {
        rx = x[i] - x[j];
        ry = y[i] - y[j];
        rz = z[i] - z[j];
        r = sqrt( (rx * rx) + (ry * ry) + (rz * rz) );
        pcs += q[i] * q[j] / r;
    }
    cp[i] = pcs;


    return;
}


int main()
{
  int i, n;
  double *x, *y, *z, *q, *cp, cs;
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
  cudaMallocManaged(&x, n * NTB * sizeof(double));
  cudaMallocManaged(&y, n * NTB * sizeof(double));
  cudaMallocManaged(&z, n * NTB * sizeof(double));
  cudaMallocManaged(&q, n * NTB * sizeof(double));
  cudaMallocManaged(&cp, n * NTB * sizeof(double));
  printf("N=%-12dNTB=%-12dn=%-12d\n", N, NTB, n);

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
  * fill the final section of individual arrays with dummy data
  */
  for (i = N; i < n * NTB; i++) {
      x[i] = (double) rand() / (double) i;
      y[i] = (double) rand() / (double) i;
      z[i] = (double) rand() / (double) i;
      q[i] = (double) 0;
  }

 /*
  * launch a GPU kernel for the computation of individual 
  * Coulomb potentials (partial sums per particle) which later
  * still need to be added up to yield the full Coulomb sum
  * n.b. now the kernel execution configuration is a true
  *      2-level one, consisting of a bunch of threadblocks
  *      forming the blockgrid;
  */
  coulomb<<<n, NTB>>>(n * NTB, &x[0], &y[0], &z[0], &q[0], &cp[0]);
  cudaDeviceSynchronize(); 

 /*
  * computing the full Coulomb sum now is really a minor effort  
  * adding together all potentials already computed by the kernel
  */
  cs = (double) 0;
  for (i = 0; i < N; i++) {
      cs += cp[i];
  }
  printf("coulomb sum: %14.9lf\n", cs);


 /*
  * and not to forget, free the allocated memory 
  */
  cudaFree(cp);
  cudaFree(q);
  cudaFree(z);
  cudaFree(y);
  cudaFree(x);


  return 0;
}
