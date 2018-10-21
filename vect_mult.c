#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> 

void read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, 
      MPI_Comm comm);
void read_data(double local_vec1[], double local_vec2[], double* scalar_p,
      int local_n, int my_rank, int comm_sz, MPI_Comm comm);
void print_vector(double local_vec[], int local_n, int n, char title[], 
      int my_rank, MPI_Comm comm);
double par_dot_product(double local_vec1[], double local_vec2[], 
      int local_n, MPI_Comm comm);
void par_vector_scalar_mult(double local_vec[], double scalar, 
      double local_result[], int local_n);

int main(void) {
   int n, local_n;
   double *local_vec1, *local_vec2;
   double scalar;
   double *local_scalar_mult1, *local_scalar_mult2;
   double dot_product;
   int comm_sz, my_rank;
   MPI_Comm comm;
   
   MPI_Init(NULL, NULL);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);
   
   read_n(&n, &local_n, my_rank, comm_sz, comm);
   
   local_vec1 = malloc(local_n*sizeof(double));
   local_vec2 = malloc(local_n*sizeof(double));
   local_scalar_mult1 = malloc(local_n*sizeof(double));
   local_scalar_mult2 = malloc(local_n*sizeof(double));
   
   read_data(local_vec1, local_vec2, &scalar, local_n, my_rank, comm_sz, comm);
   
   /* Print input data */
   if (my_rank == 0)
      printf("\n\n ===== input data =====\n");
   print_vector(local_vec1, local_n, n, "first vector is", my_rank, comm);
   print_vector(local_vec2, local_n, n, "second vector is", my_rank, comm);
   if (my_rank == 0){
      printf("scalar is %f\n",scalar);
   }
   
   /* Print results */

   if (my_rank ==0)
      printf("\n\n ===== result =====\n");

   /* Compute and print dot product */

   dot_product = par_dot_product(local_vec1, local_vec2, local_n, comm);
   if (my_rank == 0) {
      printf("Dot product is %f\n", dot_product);
   }
   
   /* Compute scalar multiplication and print out result */

   par_vector_scalar_mult(local_vec1, scalar, local_scalar_mult1, local_n);
   par_vector_scalar_mult(local_vec2, scalar, local_scalar_mult2, local_n);
   print_vector(local_scalar_mult1, local_n, n, 
         "product of the first vector with scalar is", 
         my_rank, comm);
   print_vector(local_scalar_mult2, local_n, n, 
         "product of the second vector with scalar is", 
         my_rank, comm);
   
   free(local_scalar_mult2);
   free(local_scalar_mult1);
   free(local_vec2);
   free(local_vec1);

   MPI_Finalize();
   return 0;
}


void read_n(int* n_p, int* local_n_p, int my_rank, int comm_sz, MPI_Comm comm) {
   
   if (my_rank == 0){
      printf("What is the order of the vector?\n");
      scanf("%d", n_p);
   }
   /* TO BE FILLED (Give n value to all processors) */
   MPI_Bcast(n_p , 1, MPI_INTEGER, 0, comm);

   *local_n_p = *n_p / comm_sz;
}


void read_data(double local_vec1[], double local_vec2[], double* scalar_p,
      int local_n, int my_rank, int comm_sz, MPI_Comm comm) {
   double* a = NULL;
   int i;
   int root =0;
   if (my_rank == 0){
      printf("What is the scalar?\n");
      scanf("%lf", scalar_p);
   }
   
   /* TO BE FILLED (Give scalar_p value to all processors) */
   MPI_Bcast(&scalar_p , 1, MPI_DOUBLE, 0, comm);
   
   if (my_rank == 0){
      a = malloc(local_n * comm_sz * sizeof(double));
      printf("Enter the first vector\n");
	  /* TO BE FILLED (Get first vector from user and scatter to processors) */
     for(i =0; i < local_n * comm_sz ; i++ )
	scanf("%lf", &a[i]);
	

     MPI_Scatter( a, local_n, MPI_DOUBLE, local_vec1, local_n, MPI_DOUBLE, root, comm); 
     printf("Enter the second vector\n");
	  /* TO BE FILLED (Get second vector from user and scatter to processors) */
     for(i =0;i < local_n * comm_sz; i++ )
	scanf("%lf", &a[i]);
	
     MPI_Scatter( a, local_n, MPI_DOUBLE, local_vec2, local_n, MPI_DOUBLE, root, comm); 

      free(a);
   } else {
	  /* TO BE FILLED (Scatter for vector1 and vector2) */
	MPI_Scatter( a, local_n, MPI_DOUBLE, local_vec1, local_n, MPI_DOUBLE, root, comm); 
	MPI_Scatter( a, local_n, MPI_DOUBLE, local_vec2, local_n, MPI_DOUBLE, root, comm); 

   }
}


void print_vector(double local_vec[], int local_n, int n, char title[], int my_rank, MPI_Comm comm) {
   double* a = NULL;
   int i;
   
   if (my_rank == 0) {
      a = malloc(n * sizeof(double));
	  /* TO BE FILLED (Collect scattered local_vec to a) */
	MPI_Gather(local_vec, local_n, MPI_DOUBLE, a, local_n, MPI_DOUBLE, 0, comm);
      printf("%s\n", title);
      for (i = 0; i < n; i++) 
         printf("%.2f ", a[i]);
      printf("\n");
      free(a);
   } else {
	  /* TO BE FILLED (Collect scattered local_vec to a) */
        MPI_Gather(local_vec, local_n, MPI_DOUBLE, a, local_n, MPI_DOUBLE, 0, comm);
   }

}


double par_dot_product(double local_vec1[], double local_vec2[], 
      int local_n, MPI_Comm comm) {
   int local_i;
   double dot_product=0, local_dot_product = 0;

   /* TO BE FILLED (Calculate local_dot_product using local_vec1 & local_vec2. Then, get dot_product.) */
	for(local_i =0 ; local_i < local_n; local_i++) 
		local_dot_product += local_vec1[local_i]*local_vec2[local_i];
   MPI_Reduce(&local_dot_product , &dot_product, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
   return dot_product;
}


void par_vector_scalar_mult(double local_vec[], double scalar, 
      double local_result[], int local_n) {
   int local_i;

   /* TO BE FILLED (Calculate local_result using local_vec and scalar) */
	for(local_i =0 ; local_i < local_n; local_i++) 
		local_result[local_i] = local_vec[local_i] * scalar;
}
