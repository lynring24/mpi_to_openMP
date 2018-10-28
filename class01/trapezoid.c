#include <stdio.h>
#include <string.h>	//For strlen
#include <mpi.h>	//For MPI functions, etc


const int MAX_STRING = 100;
double Trap (double a, double b, int c, double d);
double f(double x);
int main(void) {
	int comm_sz;	//number of processes
	int my_rank;	//my process rank
	int n = 1024, local_n;
	double a = 0.0, b=3.0, h, local_a, local_b;
	double local_int, total_int;
	int source;
	double start, finish, elapsed, local_elapsed;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Barrier(MPI_COMM_WORLD);
	

	h = (b-a)/n;
	local_n = n/comm_sz;
	local_a = a + my_rank*local_n*h;
	local_b = local_a + local_n*h;
	start = MPI_Wtime();
	local_int = Trap(local_a, local_b, local_n, h);
	if(my_rank != 0) {
		MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0,
				MPI_COMM_WORLD);
		
	} else {
		total_int = local_int;
		for(source = 1 ; source < comm_sz; source++) {
			MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0,
				       	MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			total_int += local_int;
		}
	}
	finish = MPI_Wtime();
	local_elapsed = finish - start;
	printf("rank : %d, local elapsed time : %e \n", my_rank, local_elapsed);
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0 , MPI_COMM_WORLD);
	if (my_rank ==0){
		printf("=========================================\n");
		printf("With n = %d trapezoids, our estimate \n", n);
		printf("of the integral from %f to %f = %.15e\n", a, b, total_int);
		printf("Elapsed time = %e seconds\n", elapsed);
	}
	MPI_Finalize();
	return 0;
}

double Trap(
		double left_endpt,
		double right_endpt,
		int trap_count,
		double base_len) {
	double estimate, x;
	int i;

	estimate = (f(left_endpt) + f(right_endpt))/2.0;
	for(i = 1; i<=trap_count-1; i++ ) {
		x = left_endpt + i* base_len;
		estimate += f(x);
	}
	estimate = estimate*base_len;
	return estimate;
}

double f(double x) {
	return x*x - 3*x;
}
