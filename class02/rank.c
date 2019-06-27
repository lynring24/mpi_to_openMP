#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(void) {
	int my_rank, comm_sz;
	const int MAX_STRING =100;
	char msg[100];

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	if(my_rank==0) {
	printf("Proc %d of %d > Does anyone have a toothpick?\n", my_rank, comm_sz);
	for(int i=1; i<comm_sz; i++){
		MPI_Recv(msg,MAX_STRING, MPI_CHAR,i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("%s\n", msg);
	}
	}
	else {
		sprintf(msg, "Proc %d of %d > Does anyone have a toothpick?\n", my_rank, comm_sz);
		MPI_Send(msg, strlen(msg)+1, MPI_CHAR, 0,0, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}
