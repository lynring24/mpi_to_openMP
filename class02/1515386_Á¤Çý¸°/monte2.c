#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
long long int Monte_carlo(long long local_number_of_tosses,int my_rank);
int main(void) {
	long long int number_of_tosses = 0;
	long long int local_number_of_tosses = 0;	
	int comm_sz, my_rank;
	double pi_estimate, local_pi_estimate;
	double start, finish, elapsed, local_elapsed;

	int number_in_circle = 0;
	
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	if(my_rank==0) {
		printf("Enter the total number of tosses\n");
		scanf("%lld", &number_of_tosses);
		local_number_of_tosses = number_of_tosses / comm_sz;
		
	}
	MPI_Bcast(&local_number_of_tosses , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();

	number_in_circle = Monte_carlo(local_number_of_tosses, my_rank);

	local_pi_estimate =  4*number_in_circle / ((double) local_number_of_tosses);

	finish = MPI_Wtime();
	local_elapsed = finish - start;
	printf("rank : %d, local elapsed time : %e \n", my_rank, local_elapsed);

	MPI_Reduce(&local_pi_estimate, &pi_estimate, 1, MPI_DOUBLE, MPI_MAX, 0 , MPI_COMM_WORLD);
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0 , MPI_COMM_WORLD);
	
	if (my_rank ==0){
		printf("==========================\n");
		printf("pi_estimate = %f \n", pi_estimate);
		printf("Elapsed time = %e seconds \n", elapsed);
	}
	MPI_Finalize();
        return 0;	
}



long long int Monte_carlo(long long number_of_tosses, int my_rank) {
	double x, y, distance_squared;
	int number_in_circle = 0;

	for(int toss=0; toss <number_of_tosses; toss++) {
		x = (double)rand()/RAND_MAX*2.0-1.0;
		y = (double)rand()/RAND_MAX*2.0-1.0;

		distance_squared = x*x+y*y;
		if(distance_squared <=1 )
			number_in_circle++;
	}
	
	return number_in_circle;
}
