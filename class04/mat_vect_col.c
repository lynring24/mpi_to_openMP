#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void Get_dims(int* m_p, int* local_m_p, int* n_p, int* local_n_p,
      int my_rank, int comm_sz, MPI_Comm comm);
void Allocate_arrays(double** local_A_pp, double** local_x_pp, 
      double** local_y_pp, int m, int local_m, int local_n, 
      MPI_Comm comm);
void Build_derived_type(int m, int local_m, int n, int local_n,
      MPI_Datatype* block_col_mpi_t_p);
void Read_matrix(char prompt[], double local_A[], int m, int local_n, 
      int n, MPI_Datatype block_col_mpi_t, int my_rank, MPI_Comm comm);
void Print_matrix(char title[], double local_A[], int m, int local_n, 
      int n, MPI_Datatype block_col_mpi_t, int my_rank, MPI_Comm comm);
void Read_vector(char prompt[], double local_vec[], int n, int local_n, 
      int my_rank, MPI_Comm comm);
void Print_vector(char title[], double local_vec[], int n,
      int local_n, int my_rank, MPI_Comm comm);
void Mat_vect_mult( double local_A[], double local_x[], double local_y[], int local_m , int  m,
               int n, int local_n, int comm_sz, int local_rank,	MPI_Comm  comm ) ;
char fileName[10];
FILE *fp;

int main(void) {
   double* local_A;
   double* local_x;
   double* local_y;
   
   int m,n;
   int local_m, local_n;
   double local_start, local_finish, local_elapsed, elapsed;

   int my_rank, comm_sz;
   MPI_Comm comm;
   MPI_Datatype block_col_mpi_t;
   
   MPI_Init(NULL, NULL);
   comm = MPI_COMM_WORLD;
   MPI_Comm_size(comm, &comm_sz);
   MPI_Comm_rank(comm, &my_rank);
   
   Get_dims(&m, &local_m, &n, &local_n, my_rank, comm_sz, comm);
   Allocate_arrays(&local_A, &local_x, &local_y, m, local_m, local_n, comm);
   Build_derived_type(m, local_m, n, local_n, &block_col_mpi_t);
   Read_matrix("A", local_A, m, local_n, n, block_col_mpi_t, my_rank, comm);
   Print_matrix("A", local_A, m, local_n, n, block_col_mpi_t, my_rank, comm);
   
   Read_vector("x", local_x, n, local_n, my_rank, comm);
   Print_vector("x", local_x, n, local_m, my_rank, comm);
   
   local_start = MPI_Wtime();
   Mat_vect_mult(local_A, local_x, local_y, local_m, m, n, local_n, comm_sz, my_rank,  comm);   
   local_finish = MPI_Wtime();
   local_elapsed = local_finish - local_start;
   printf("rank : %d, local elapsed time : %e\n", my_rank, local_elapsed);
 
   MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
   if(my_rank == 0)
	   printf("Elapsed time = %e seconds\n", elapsed);

   Print_vector("y", local_y, m, local_m, my_rank, comm);

   free(local_A);
   free(local_x);
   free(local_y);
   //MPI_Type_free(&block_col_mpi_t);
   MPI_Finalize();
   return 0;
}
//https://stackoverflow.com/questions/4600797/read-int-values-from-a-text-file-in-c?answertab=votes#tab-top
void Get_dims(
      int*      m_p        /* out */, 
      int*      local_m_p  /* out */,
      int*      n_p        /* out */,
      int*      local_n_p  /* out */,
      int       my_rank    /* in  */,
      int       comm_sz    /* in  */,
      MPI_Comm  comm       /* in  */) {

   if (my_rank == 0) {
      printf("Enter the file name\n");
	  
	  /* TO BE FILLED (Get file name and open it. Get m_p(=n_p) value) */
     scanf("%s", fileName);
     fp = fopen (fileName, "r");
     fscanf(fp, "%d", m_p);
   }
   /* TO BE FILLED (give m_p value to every processor) */
     MPI_Bcast(m_p , 1, MPI_INTEGER, 0, comm);
   *n_p = *m_p;
   *local_m_p = *m_p/comm_sz;
   *local_n_p = *n_p/comm_sz;
}

void Allocate_arrays(
   double**  local_A_pp  /* out */, 
   double**  local_x_pp  /* out */, 
   double**  local_y_pp  /* out */, 
   int       m           /* in  */,   
   int       local_m     /* in  */, 
   int       local_n     /* in  */, 
   MPI_Comm  comm        /* in  */) {
   
   *local_A_pp = malloc(m*local_n*sizeof(double));
   *local_x_pp = malloc(local_n*sizeof(double));
   *local_y_pp = malloc(local_m*sizeof(double));
}

void Build_derived_type(int m, int local_m, int n, int local_n,
      MPI_Datatype* block_col_mpi_t_p) {
   MPI_Datatype vect_mpi_t;

   /* m blocks each containing local_n elements */
   /* The start of each block is n doubles beyond the preceding block */
   MPI_Type_vector(m, local_n, n, MPI_DOUBLE, &vect_mpi_t);

   /* Resize the new type so that it has the extent of local_n doubles */
   MPI_Type_create_resized(vect_mpi_t, 0, local_n*sizeof(double),
         block_col_mpi_t_p);
   MPI_Type_commit(block_col_mpi_t_p);
}

void Read_matrix(
             char          prompt[]         /* in  */, 
             double        local_A[]        /* out */, 
             int           m                /* in  */, 
             int           local_n          /* in  */, 
             int           n                /* in  */,
             MPI_Datatype  block_col_mpi_t  /* in  */,
             int           my_rank          /* in  */,
             MPI_Comm      comm             /* in  */) {
   double* A = NULL;

   if (my_rank == 0) {
      A = malloc(m*n*sizeof(double));
      printf("\nThe matrix A\n");

	  /* TO BE FILLED (Get matrix A from file and print it. Scatter it to all processors. 
	  When you use scatter, the send datatype is block_col_mpi_t, recieve datatype is MPI_DOUBLE) */
	for(int i=0; i < m; i++ )
 	   for( int j=0; j<n; j++) 
		fscanf(fp, "%lf", &A[i*n+j]);
       MPI_Scatter(A,  local_n*m, MPI_DOUBLE, local_A, local_n*m, MPI_DOUBLE, 0, comm);
       free(A);
   } else {
      /* TO BE FILLED (Scatter)*/
      MPI_Scatter(A,  local_n*m, MPI_DOUBLE, local_A, local_n*m, MPI_DOUBLE, 0, comm);
   }
}

void Print_matrix(char title[], double local_A[], int m, int local_n, 
      int n, MPI_Datatype block_col_mpi_t, int my_rank, MPI_Comm comm) {
	double* A = NULL;
   int i, j;

   if (my_rank == 0) {
      A = malloc(m*n*sizeof(double));

	  /* TO BE FILLED (Collect values from processors and print it. 
	  Send datatype is MPI_DOUBLE, and recieve datatype is block_col_mpi_t.) */


      MPI_Gather(local_A, m*local_n, MPI_DOUBLE, A, m*local_n, MPI_DOUBLE, 0, comm);
	
	for(i=0; i<m ; i++) {
		for(j=0;j<n;j++) 
			printf("%lf ", A[i*n+j]);
		printf("\n");
	}

      free(A);
   } else {
	  /* TO BE FILLED (Collect values from processors) */
	MPI_Gather(local_A, m*local_n, MPI_DOUBLE, A, m*local_n, block_col_mpi_t, 0, comm);

   }
}

void Read_vector(
             char      prompt[]     /* in  */, 
             double    local_vec[]  /* out */, 
             int       n            /* in  */,
             int       local_n      /* in  */,
             int       my_rank      /* in  */,
             MPI_Comm  comm         /* in  */) {
   double* vec = NULL;
   int root = 0;
   int i=0;
    if (my_rank == 0) {
      vec = malloc(n*sizeof(double));
      // printf("\nThe vector %s\n", prompt);

	  /* TO BE FILLED (Get value from the file and print it. 
	  Scatter the values. Send datatype and Recieve datatype is MPI_DOUBLE.)*/
	for(i =0; i<n; i++) 
	    fscanf(fp, "%lf", &vec[i]);
	MPI_Scatter(vec, local_n, MPI_DOUBLE, local_vec, local_n, MPI_DOUBLE, root, comm); 
      free(vec);
   } else {
	   /* TO BE FILLED (Scatter)*/
	MPI_Scatter(vec, local_n, MPI_DOUBLE, local_vec, local_n, MPI_DOUBLE, root, comm); 
   }
}


void Mat_vect_mult(
               double    local_A[]  /* in  */, 
               double    local_x[]  /* in  */, 
               double    local_y[]  /* out */,
               int       local_m    /* in  */, 
               int       m          /* in  */,
               int       n         /* in  */,
               int       local_n    /* in  */, 
               int       comm_sz,
               int       local_rank,	
               MPI_Comm  comm       /* in  */) {
   
   double* my_y;
   double* x;
   int* recv_counts;
   int i, j;
   int m_loc; 

   x = malloc(n*sizeof(double));

   my_y = malloc(n*sizeof(double));
   for( j=0 ; j < n; j++) 
	my_y[j] = 0;

   MPI_Allgather(local_x, local_n, MPI_DOUBLE, x, local_n,MPI_DOUBLE, comm);

/*test code
 printf("rank %d vector x\n", local_rank);	
  for( j=0 ; j < n; j++) 
	printf("%lf ", x[j]);
  printf("\n");	*/

 /* TO BE FILLED (Use local_a and local_x to multiply and add it to my_y)
  for (i=0; i < local_m; i++) {
	my_y[i]=0;
	for( j=0 ; j < n; j++) {
		// printf("[%d,%d][%lf, %lf]\n", i,j,local_A[i*n+j], x[j]);
		my_y[i] += local_A[i*n+j] * x[j];
	}
   } */


// save the result in right location, the location should be local_start = rank * unit_size + start to local_end = start + unit_size 
   m_loc =  local_rank * local_m; 	
    for (i=0; i < local_m; i++) {
	for( j=0 ; j < n; j++) {
		my_y[i+m_loc] += local_A[i*n+j] * x[j];
	}
   } 
  
 printf("rank %d vector y\n", local_rank);	
  for( j=0 ; j < n; j++) 
	printf("%lf ", my_y[j]);
  printf("\n");



  recv_counts = malloc(comm_sz*sizeof(int));
   
  for (i = 0; i < comm_sz; i++) 
      recv_counts[i] = local_m;
   
   MPI_Reduce_scatter(my_y, local_y, recv_counts, MPI_DOUBLE, MPI_SUM, comm);
   free(x);
   free(my_y);
   free(recv_counts);
}

void Print_vector(
              char      title[]     /* in */, 
              double    local_vec[] /* in */, 
              int       n           /* in */,
              int       local_n     /* in */,
              int       my_rank     /* in */,
              MPI_Comm  comm        /* in */) {
   double* vec = NULL;

   if (my_rank == 0) {
      printf("\nThe vector %s\n", title);
      vec = malloc(n*sizeof(double));
	  /* TO BE FILLED (Collect local_vec to vec and print it. Send datatype and recieve datatype is MPI_DOUBLE.)*/
      MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0, comm);
	
      for(int i=0; i<n; i++) 
	printf("%lf ", vec[i]);
      printf("\n");

      free(vec);
   }  else {
	  /* TO BE FILLED (Collect local_vec to vec)*/
      MPI_Gather(local_vec, local_n, MPI_DOUBLE, vec, local_n, MPI_DOUBLE, 0, comm);
   }
}
