#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define N 1024

long long numbers[N];

typedef struct{
    long long start_index; //start index of current thread
    long long end_index; //end index of current thread
    long long partialSum; //partial sum calculated by current thread
}ThreadData;

void* sum_helper(void* arg){
    ThreadData* data = (ThreadData*)arg; //typecasting arg to thread_data
    long long sum = 0;

    for(long long i = data->start_index; i<= data->end_index; i++){
        sum += numbers[i];
    }
    data->partialSum = sum;
    pthread_exit(NULL);
}

int main(int argc, char **argv){
    int num_threads = atoi(argv[1]);

    for(int i = 0; i< N;i++){
        numbers[i] = i + 1;
    }
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    pthread_t threads[num_threads]; //holds the unique identifiers the OS gives to each of the threads u create
    ThreadData thread_data[num_threads]; //holds the arguments to pass into each of the thread

    int partition_size = N / num_threads;
    int start_idx = 0;

    //threads are summing the values individually.....
    for(int i = 0; i < num_threads; i++){
        thread_data[i].start_index = start_idx;
        if(i == num_threads - 1){
            thread_data[i].end_index = N - 1;
        }else{
            thread_data[i].end_index = start_idx + partition_size - 1;
        }
        thread_data[i].partialSum = 0;
        if(pthread_create(&threads[i],NULL,sum_helper,&thread_data[i])  != 0){
            perror("Failed to create thread");
            return 1;
        }
        start_idx += partition_size;
    }

    //join the partial sum within each thread....
    long long totalSum = 0;
    for(int i = 0; i< num_threads;i++){
        if(pthread_join(threads[i],NULL) != 0){
            perror("Failed to join threads");
            return 1;
        }
        totalSum += thread_data[i].partialSum;
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1e6;
    elapsed_time += (end_time.tv_nsec - start_time.tv_nsec) / 1e3;

    printf("Summing array of size %d using %d threads.\n", N, num_threads);
    printf("Total sum: %lld\n", totalSum);
    printf("Calculation took %.2f microseconds.\n", elapsed_time);

    return 0;
}

/*
Assignment 1 done by Shashank Shashidhar IIT2022008 
Experimental configurations:
CPU architecture: x86_64
Address sizes: 39 bits physical, 48 bits virtual
Logical CPUs: 8
Cores per socket: 4
Threads per core: 2
Socket: 1
Model name: Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz

Execution times of above program:
1 thread: 1062.27 microseconds.
2 threads: 1113.69 microseconds.
4 threads: 2565.36 microseconds.
8 threads: 2694.90 microseconds.
16 threads: 3530.06 microseconds.

It is very counter intuitive that having more number of threads is resulting in an increase in execution of time.
Clearly if i have more threads, it should be able to execute a serial program faster?
This is not so and we see an increase in execution time with increase in threads because there is the added overhead of handling, managing and merging the created threads.
The OS has an added responsibility of executing these threads concurrently and then merging them.
This overhead increases time taken to finish executing.
Multithreading is only actually beneficial when the task being executed by the thread is significantly greater than the overhead of handling and joining them.

*/