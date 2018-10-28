#include <stdio.h>
#include <mpi.h>
#include <string.h>
#define parter(x) (x+1)%2

int MAXCOUNT = 19;

int main(void) {
	int my_rank, comm_sz;
	double start, finish;
	int msg = -1, count = 0;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Barrier(MPI_COMM_WORLD);
	while(count < MAXCOUNT) {
	 if (count %2 == my_rank) {
		start = MPI_Wtime();	
		count++;
		if(msg<1) 
			msg= msg+1;
		else 
			msg=msg*2;
		MPI_Send(&count, 1, MPI_INT,parter(my_rank), 0, MPI_COMM_WORLD);
		MPI_Send(&msg, 1, MPI_INT, parter(my_rank), 0, MPI_COMM_WORLD);
		}
	 else {
		MPI_Recv(&count, 1, MPI_INT, parter(my_rank), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       		MPI_Recv(&msg, 1, MPI_INT, parter(my_rank), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		finish = MPI_Wtime();	
		printf("%d %e \n", msg, (finish-start)/2);
	 }
	}
	MPI_Finalize();
	return 0;
}


