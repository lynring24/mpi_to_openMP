#include<stdio.h>
#include<time.h>
#include<stdlib.h>

int main(void) {
	int number_in_circle = 0;
	long long int number_of_tosses = 0;	
	double x, y, distance_squared;
	double pi_estimate;
	//time variables
	clock_t start, end;
	double cpu_time_used;

	printf("Enter the total number of tosses\n");
	scanf("%lld", &number_of_tosses);

	start = clock();
	for(int toss=0; toss <number_of_tosses; toss++) {
		x = (double)rand()/RAND_MAX*2.0-1.0;
		y = (double)rand()/RAND_MAX*2.0-1.0;

		distance_squared = x*x+y*y;
		if(distance_squared <=1 )
			number_in_circle++;
	}
	
	pi_estimate = (4*number_in_circle) / ((double) number_of_tosses);
	end = clock();
	cpu_time_used = ((double)(end - start)) /CLOCKS_PER_SEC;
	printf("pi_estimate = %f \n", pi_estimate);
	printf("Elapsed time = %f seconds \n", cpu_time_used);
        return 0;	
}
