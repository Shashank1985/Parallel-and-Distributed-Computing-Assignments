/*
Parallel and Distributed Computing Assignment 2, done by Shashank Shashidhar IIT2022008

Counting prime numbers from 1 to 2^n where n is the maximum possible n on my system (n = 32)
Experimental Setup:
CPU architecture: x86_64
CPU op-mode(s):         32-bit, 64-bit
Address sizes:          39 bits physical, 48 bits virtual
CPU(s): 8 with 4 cores and 2 threads per core. 
Compilation of code is done on WSL, using gcc compiler with -pthread and -lm flag

Results:
Enter the value of n: 32
Calculating primes up to 2^32 = 4294967296.


 Starting Serial Prime Count
Serial Execution Time: 84.845903 seconds
Number of primes found: 203280221

 Starting Parallel Count
Parallel Execution Time: 66.440402 seconds
Number of primes found: 203280221

We can see that from 1 to 2 ^ 32, we have 203280221 prime numbers and parallel execution will lead to a time decrease
of nearly 21.4%. The serial counting of the prime numbers took 84.845903 seconds and the parallel execution with 4 threads
took 66.440402 seconds

The counting of prime numbers was achieved using the sieve of eratosthenes algorithm. We use a large char array of size
2 ^ 32 where 1 represents the integer at that position is prime, 0 otherwies. The serial execution carries out this algorithm.
In parallel execution, there is a subtle change. When the above approach using only 1 array was replicated for 
parallel execution, the time taken would increase as multiple threads were writing to the same array. This led to a kind
of traffic jam between the threads, where each thread was waiting for the other to finish to writing to the array.

To fix this, we provided a seperate array for each thread and each thread would write to that array and finally the result
would be aggregated over all the threads. This ensures the core principle of "maximizing independent writes and minimizing shared writes"
*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define NUM_THREADS 4

unsigned long long limit;
unsigned long long total_prime_count = 0;
unsigned long long *base_primes;
int num_base_primes = 0;

typedef struct {
    int id;
    unsigned long long local_count;
} ThreadData;

void countPrimesSerial();
void countPrimesParallel();
void display(); 
void *sieve_worker(void *arg);

int main() {
    int n;
    printf("Enter the value of n: ");
    scanf("%d", &n);

    limit = 1ULL << n; // 2^n
    printf("Calculating primes up to 2^%d = %llu.\n\n", n, limit);

    countPrimesSerial();
    countPrimesParallel();

    return 0;
}

void countPrimesSerial() {
    printf("\n Starting Serial Prime Count \n");
    struct timespec start, end;
    
    char *isPrime = (char *)malloc((limit + 1) * sizeof(char));
    if (isPrime == NULL) {
        printf("Serial: Failed to allocate memory.\n");
        return;
    }
    memset(isPrime, 1, limit + 1);
    isPrime[0] = isPrime[1] = 0;

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (unsigned long long p = 2; p * p <= limit; p++) {
        if (isPrime[p] == 1) {
            for (unsigned long long i = p * p; i <= limit; i += p) {
                isPrime[i] = 0;
            }
        }
    }

    unsigned long long count = 0;
    for (unsigned long long p = 2; p <= limit; p++) {
        if (isPrime[p] == 1) {
            count++;
        }
    }
    total_prime_count = count;

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Serial Execution Time: %f seconds\n", time_taken);
    printf("Number of primes found: %llu\n", total_prime_count);

    free(isPrime);
}

void countPrimesParallel() {
    printf("\n Starting Parallel Count \n");
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    unsigned long long limit_sqrt = (unsigned long long)sqrt(limit);
    char *sqrt_sieve = (char *)malloc((limit_sqrt + 1) * sizeof(char));
    if (sqrt_sieve == NULL) {
        printf("Parallel: Failed to allocate memory for base sieve.\n");
        return;
    }
    memset(sqrt_sieve, 1, limit_sqrt + 1);
    sqrt_sieve[0] = sqrt_sieve[1] = 0;

    for (unsigned long long p = 2; p * p <= limit_sqrt; p++) {
        if (sqrt_sieve[p] == 1) {
            for (unsigned long long i = p * p; i <= limit_sqrt; i += p) {
                sqrt_sieve[i] = 0;
            }
        }
    }

    unsigned long long base_count = 0;
    for (unsigned long long p = 2; p <= limit_sqrt; p++) {
        if (sqrt_sieve[p] == 1) {
            base_count++;
        }
    }
    base_primes = (unsigned long long *)malloc(base_count * sizeof(unsigned long long));
    if (base_primes == NULL) {
        printf("Parallel: Failed to allocate memory for base primes array.\n");
        free(sqrt_sieve);
        return;
    }
    num_base_primes = 0;
    for (unsigned long long p = 2; p <= limit_sqrt; p++) {
        if (sqrt_sieve[p] == 1) {
            base_primes[num_base_primes++] = p;
        }
    }
    free(sqrt_sieve); 
    total_prime_count = base_count;

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].id = i;
        thread_data[i].local_count = 0;
        if (pthread_create(&threads[i], NULL, sieve_worker, &thread_data[i]) != 0) {
            perror("Failed to create thread");
            return;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_prime_count += thread_data[i].local_count;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    double time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    printf("Parallel Execution Time: %f seconds\n", time_taken);
    printf("Number of primes found: %llu\n", total_prime_count);
    
    free(base_primes);
}

void *sieve_worker(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int thread_id = data->id;

    unsigned long long limit_sqrt = (unsigned long long)sqrt(limit);
    
    unsigned long long start_num = limit_sqrt + 1;
    unsigned long long range = limit - start_num + 1;
    unsigned long long block_size = range / NUM_THREADS;
    
    unsigned long long block_start = start_num + thread_id * block_size;
    unsigned long long block_end = (thread_id == NUM_THREADS - 1) ? limit : block_start + block_size - 1;

    char *block_sieve = (char *)malloc(block_size * sizeof(char));
    if(block_sieve == NULL) {
        printf("Thread %d: Failed to allocate block memory.\n", thread_id);
        pthread_exit(NULL);
    }
    memset(block_sieve, 1, block_size);

    for (int i = 0; i < num_base_primes; i++) {
        unsigned long long p = base_primes[i];
        
        unsigned long long first_multiple = (block_start / p) * p;
        if (first_multiple < block_start) {
            first_multiple += p;
        }
        
        for (unsigned long long j = first_multiple; j <= block_end; j += p) {
            block_sieve[j - block_start] = 0;
        }
    }

    for (unsigned long long i = 0; i < (block_end - block_start + 1); i++) {
        if (block_sieve[i] == 1) {
            data->local_count++;
        }
    }
    
    free(block_sieve);
    pthread_exit(NULL);
}
