#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "timer.h"

int thread_count;
void* generateFunc(void* rank);

int main(int argc, char* argv[]) {
	long thread;
	double start, end, elapsed;
	pthread_t* thread_handles;
	
	thread_count = strtol(argv[1], NULL, 10);
	
	thread_handles = malloc(thread_count * sizeof(pthread_t));
	
	//start dividing thread
	GET_TIME(start);

	for(thread = 0; thread < thread_count; thread++)
		pthread_create(&thread_handles[thread], NULL, generateFunc, (void*)thread);
	
	for(thread = 0; thread < thread_count; thread++)
		pthread_join(thread_handles[thread], NULL);
	
	GET_TIME(end);

	elapsed = end - start;
	printf("Runtime for creating and terminating %d threads : %e\n ", thread_count, elapsed/thread_count);

	free(thread_handles);
	return 0;
}

void* generateFunc (void* rank) {
	return NULL;
}
