#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "timer.h"
#include <semaphore.h>
#include <math.h>

int thread_count;
int counter=0;
void* mutex_pi (void* rank);
void* semaphore_pi (void* rank);
double sum=0;
int n;
pthread_mutex_t mutex;
sem_t sem;

int main(int argc, char* argv[]) {
	long thread;
	double start, end, elapsed;
	pthread_t* thread_handles;
	
	thread_count = strtol(argv[1], NULL, 10);
	n = strtol(argv[2], NULL, 10);
	printf("With n = %d terms, \n", n);
	thread_handles = malloc(thread_count * sizeof(pthread_t));		

	//semaphore
	sem_init(&sem, 0, 1);
	GET_TIME(start);
	for(thread = 0; thread < thread_count; thread++)
		pthread_create(&thread_handles[thread], NULL, semaphore_pi, (void*)thread);
	
	for(thread = 0; thread < thread_count; thread++)
		pthread_join(thread_handles[thread], NULL);
	GET_TIME(end);

	elapsed = end - start;
	
	printf("\tSemaphore estimate of pi = %g\n, The elapsed time is %e seconds \n", 4*sum, elapsed);
	sem_destroy(&sem);

	// mutex
	sum = 0 ;
	pthread_mutex_init(&mutex, NULL);
	GET_TIME(start);
	for(thread = 0; thread < thread_count; thread++)
		pthread_create(&thread_handles[thread], NULL, mutex_pi, (void*)thread);
	
	for(thread = 0; thread < thread_count; thread++)
		pthread_join(thread_handles[thread], NULL);
	
	GET_TIME(end);
	elapsed = end - start;
	printf("\tMutex estimate of pi = %g\n The elapsed time is %e seconds \n", 4*sum, elapsed);
	
	pthread_mutex_destroy(&mutex);

	//actual result
	printf("\t\t\t pi = %lf\n ",  4*atan(1.0));
	free(thread_handles);
	return 0;
}

void* semaphore_pi (void* rank) {
	long my_rank = (long) rank;
	double my_sum=0;
	double factor;
	long long i;
	long long my_n = n/ thread_count;
	long long my_first_i = my_n * my_rank;
	long long my_last_i = my_first_i + my_n;
	
	if ( my_first_i % 2 == 0 )
		factor = 1.0;
	else
		factor = -1.0;
	
	for (i = my_first_i; i < my_last_i; i++, factor = -factor) 
		my_sum += factor/(2*i+1);
			
	sem_wait(&sem);
	sum += my_sum;
	sem_post(&sem);
	
	return NULL;
}




void* mutex_pi (void* rank) {
	long my_rank = (long) rank;
	double factor;
	long long i;
	long long my_n = n/ thread_count;
	long long my_first_i = my_n * my_rank;
	long long my_last_i = my_first_i + my_n;
	double my_sum=0;
	
	if ( my_first_i % 2 == 0 )
		factor = 1.0;
	else
		factor = -1.0;

	for (i = my_first_i; i < my_last_i; i++, factor = -factor) {
		my_sum += factor/(double)(2*i+1);
	}

	pthread_mutex_lock(&mutex);
	sum += my_sum;
	pthread_mutex_unlock(&mutex);

	return NULL;
}


