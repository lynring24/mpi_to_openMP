#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include "timer.h"

double Trap (double a, double b, int c, double d);
double f(double x);
void * call_option (void * rank);

const int MAX_STRING = 100;
double a = 0.0, b=3.0;
int n=0, option=0, pthread_count;
double result=0;
pthread_mutex_t mutex;
sem_t sem;
void* mutex_pi (void* rank);
void* semaphore_pi (void* rank);
int flag=0;

int main(int argc, char* argv[]) {
	double local_int;
	int thread;
	pthread_t * thread_handles;	
	double start, finish;

	pthread_count = strtol(argv[1], NULL, 10);
	option = strtol(argv[2], NULL, 10);
	thread_handles = malloc(pthread_count * sizeof(pthread_t));	
	printf("Enter a, b and n\n");
	scanf("%lf %lf %d", &a, &b,&n);
	
	GET_TIME(start);
	sem_init(&sem, 0, 1);
	pthread_mutex_init(&mutex, NULL);
	result =0;
 	for(thread = 0; thread< pthread_count; thread++)
		pthread_create(&thread_handles[thread], NULL, call_option, (void*)thread);
	
	for(thread = 0; thread < pthread_count; thread++)
		pthread_join(thread_handles[thread], NULL);
	GET_TIME(finish);
	
	sem_destroy(&sem);
	pthread_mutex_destroy(&mutex);

	printf("With n = %d trapezoids, our estimate \n", n);
	printf("of the integral from %f to %f = %.15e\n", a, b, result);
	printf("The elapsed time is %e second \n", finish - start);
	
	free(thread_handles);
	return 0;
}


void * call_option (void * rank) {

	double local_a, local_b, h;
	int local_n, local_int;
	long my_rank = (long) rank;
	double local_trap;

	h = (b-a)/n;
	local_n = n/pthread_count;
	local_a = a + my_rank * local_n * h;
	local_b = local_a + local_n*h;
	local_trap = Trap(local_a, local_b, local_n, h);
	
	switch(option) {
		case 1 : 
			pthread_mutex_lock(&mutex);
			result += local_trap;
			pthread_mutex_unlock(&mutex);
			break;
		case 2 : 
			sem_wait(&sem);
			result +=local_trap;
			sem_post(&sem);
			break;
		case 3 : 
			while(flag%pthread_count != my_rank);
			result += local_trap;
			flag = (flag+1)%pthread_count;
			break;
	}
	printf("local :%lf \n", local_trap);
	return NULL;
}

double Trap(double left_endpt, double right_endpt, int trap_count, double base_len) {
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
	return x*x;
}
