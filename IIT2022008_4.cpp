/*
Parallel and Distributed Computing Assignment 4, done by Shashank Shashidhar IIT2022008

Counting prime numbers from 1 to 2^n where n is the maximum possible n on my system (n = 32) using C++ and openmp
Experimental Setup:
CPU architecture: x86_64
CPU op-mode(s):         32-bit, 64-bit
Address sizes:          39 bits physical, 48 bits virtual
CPU(s): 8 with 4 cores and 2 threads per core. 
Compilation of code is done on WSL, using gcc compiler with -fopenmp flag

Code structure:
2 functions, one utilising parallel directive, the other using parallel + critical directive.
The algorithm is as follows:
Segmented sieve of eratosthenes. n = 32, so the limit is 2 ^ 32 which is ~4.5 billion
We first compute the primes from 2 to 2 ^ 16 (sqrt(limit)) serially, which give us our base primes. This is done serially
because 2 to 65k is easy to compute serially but the remaining part from 65k to 4.5 billion is harder to compute, so we can
parallelize that using the openmp library.
Once we have the base threads, we can split up the remaining part from 65k to 4.5 billion equally by the number of threads 
we have and have each thread compute the number of primes within the range associated to it. Then we combine the sum 
In both functions, the for directive is used to split up the range between the number of threads we have.
In parallel only directive we use reduction to ensure the addition is done properly with less time taken
In parallel + critical, the critical directive is used for serial thread addition (one by one, sort of like acquiring a lock on the variable)

Results:
Enter the value of n: 32
Enter the number of threads: 8
Calculating primes up to 2^32 = 4294967296.


Starting OpenMP Parallel Prime Count (using Reduction)
OpenMP Reduction Execution Time: 230.853 seconds
Number of primes found: 203280221
------------------------------------------

Starting OpenMP Parallel Prime Count (using Critical)
OpenMP Critical Execution Time: 263.852 seconds
Number of primes found: 203280221

We can see that reduction is faster than critical. This is because critical ensures that only one thread updates the
shared result variable at a time like acquiring a lock, while reduction is a more optimized way in which the additions are done parallely, so
although the number of additions is the same as critical, they are done parallely which reduces the time taken.
*/
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <numeric>

using namespace std;

void countPrimesOpenMP_Reduction(unsigned long long limit) {
    cout << "\nStarting OpenMP Parallel Prime Count (using Reduction)\n";
    auto start_time = chrono::high_resolution_clock::now();

    unsigned long long limit_sqrt = static_cast<unsigned long long>(sqrt(limit));
    unsigned long long total_prime_count = 0;

    vector<bool> is_prime_base(limit_sqrt + 1, true);
    is_prime_base[0] = is_prime_base[1] = false;
    for (unsigned long long p = 2; p * p <= limit_sqrt; ++p) {
        if (is_prime_base[p]) {
            for (unsigned long long i = p * p; i <= limit_sqrt; i += p) {
                is_prime_base[i] = false;
            }
        }
    }

    vector<unsigned long long> base_primes;
    for (unsigned long long p = 2; p <= limit_sqrt; ++p) {
        if (is_prime_base[p]) {
            base_primes.push_back(p);
        }
    }
    total_prime_count = base_primes.size();

    const unsigned long long block_size = limit_sqrt;
    
    #pragma omp parallel for reduction(+:total_prime_count)
    for (unsigned long long block_start = limit_sqrt + 1; block_start <= limit; block_start += block_size) {
        unsigned long long block_end = min(block_start + block_size - 1, limit);
        vector<bool> is_prime_block(block_end - block_start + 1, true);

        for (unsigned long long p : base_primes) {
            unsigned long long first_multiple = (block_start / p) * p;
            if (first_multiple < block_start) {
                first_multiple += p;
            }

            for (unsigned long long i = first_multiple; i <= block_end; i += p) {
                is_prime_block[i - block_start] = false;
            }
        }

        for (bool is_prime : is_prime_block) {
            if (is_prime) {
                total_prime_count++;
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> time_taken = end_time - start_time;
    
    cout << "OpenMP Reduction Execution Time: " << time_taken.count() << " seconds\n";
    cout << "Number of primes found: " << total_prime_count << "\n";
}

void countPrimesOpenMP_Critical(unsigned long long limit) {
    cout << "\nStarting OpenMP Parallel Prime Count (using Critical)\n";
    auto start_time = chrono::high_resolution_clock::now();

    unsigned long long limit_sqrt = static_cast<unsigned long long>(sqrt(limit));
    unsigned long long total_prime_count = 0;

    vector<bool> is_prime_base(limit_sqrt + 1, true);
    is_prime_base[0] = is_prime_base[1] = false;
    for (unsigned long long p = 2; p * p <= limit_sqrt; ++p) {
        if (is_prime_base[p]) {
            for (unsigned long long i = p * p; i <= limit_sqrt; i += p) {
                is_prime_base[i] = false;
            }
        }
    }

    vector<unsigned long long> base_primes;
    for (unsigned long long p = 2; p <= limit_sqrt; ++p) {
        if (is_prime_base[p]) {
            base_primes.push_back(p);
        }
    }
    total_prime_count = base_primes.size();

    const unsigned long long block_size = limit_sqrt;
    #pragma omp parallel
    {
        unsigned long long local_count = 0;

        #pragma omp for nowait
        for (unsigned long long block_start = limit_sqrt + 1; block_start <= limit; block_start += block_size) {
            unsigned long long block_end = min(block_start + block_size - 1, limit);
            vector<bool> is_prime_block(block_end - block_start + 1, true);

            for (unsigned long long p : base_primes) {
                unsigned long long first_multiple = (block_start / p) * p;
                if (first_multiple < block_start) {
                    first_multiple += p;
                }

                for (unsigned long long i = first_multiple; i <= block_end; i += p) {
                    is_prime_block[i - block_start] = false;
                }
            }
            
            for (bool is_prime : is_prime_block) {
                if (is_prime) {
                    local_count++;
                }
            }
        }

        #pragma omp critical
        {
            total_prime_count += local_count;
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> time_taken = end_time - start_time;
    
    cout << "OpenMP Critical Execution Time: " << time_taken.count() << " seconds\n";
    cout << "Number of primes found: " << total_prime_count << "\n";
}

int main() {
    int n, num_threads;
    cout << "Enter the value of n: ";
    cin >> n;

    cout << "Enter the number of threads: ";
    cin >> num_threads;

    unsigned long long limit = 1ULL << n;
    cout << "Calculating primes up to 2^" << n << " = " << limit << ".\n\n";
    
    omp_set_num_threads(num_threads);
    countPrimesOpenMP_Reduction(limit);
    
    cout << "------------------------------------------\n";

    omp_set_num_threads(num_threads);
    countPrimesOpenMP_Critical(limit);

    return 0;
}
