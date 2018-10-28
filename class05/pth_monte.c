#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include "timer.h"

int thread_count=0;
void *  Monte_carlo(void * rank);
long long int local_number_of_tosses = 0;
int number_in_circle = 0;

int main(int argc, char* argv[]) {
	long long int number_of_tosses = 0;
	int thread;
	double  start, finish;
	pthread_t * thread_handles;	
	double pi_estimate=0;
	
	thread_count = strtol(argv[1], NULL, 10);
	number_of_tosses = strtol(argv[2], NULL, 10);
	thread_handles = malloc(thread_count * sizeof(pthread_t));
	
	local_number_of_tosses = number_of_tosses / thread_count;	

	GET_TIME(start);	
	
	for(thread = 0; thread< thread_count; thread++)
		pthread_create(&thread_handles[thread], NULL, Monte_carlo, (void*)thread);
	
	for(thread = 0; thread < thread_count; thread++)
		pthread_join(thread_handles[thread], NULL);

	GET_TIME(finish);	

	pi_estimate =  4*number_in_circle / ((double) number_of_tosses);
	printf("pi_estimate = %f \n", pi_estimate);
	printf("Elapsed time = %e seconds \n", finish - start);

	free(thread_handles);
        return 0;	
}


void *  Monte_carlo(void * rank) {
	double x, y, distance_squared;
	int local_number_in_circle = 0;

	for(int toss=0; toss <local_number_of_tosses; toss++) {
		x = (double)rand()/RAND_MAX*2.0-1.0;
		y = (double)rand()/RAND_MAX*2.0-1.0;

		distance_squared = x*x+y*y;
		if(distance_squared <=1 )
			local_number_in_circle++;
	}
	
	
	number_in_circle += local_number_in_circle;

	return NULL;
}
