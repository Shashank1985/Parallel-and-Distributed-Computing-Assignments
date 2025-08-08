#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    // We start n from 10, as smaller values are trivial.
    // We will test up to n=64, which is the limit for unsigned long long.
    for (int n = 10; n < 64; n++) {
        // Calculate the upper bound, which is 2^n.
        // We use unsigned long long to handle very large numbers.
        unsigned long long limit = pow(2, n);

        printf("Testing for n = %d (limit = 2^%d = %llu)...\n", n, n, limit);

        // Attempt to allocate an array of 'limit' bytes (char is 1 byte).
        // This simulates the memory requirement for a Sieve of Eratosthenes.
        char *numbers = (char *)malloc(limit * sizeof(char));

        // Check if memory allocation was successful.
        if (numbers == NULL) {
            // If malloc returns NULL, it means memory allocation failed.
            printf("\n------------------------------------------------------------\n");
            printf("Memory allocation failed for n = %d.\n", n);
            printf("The largest possible 'n' for your system is likely %d.\n", n - 1);
            printf("The upper bound for your main program should be 2^%d.\n", n - 1);
            printf("------------------------------------------------------------\n");
            
            // Exit the loop and program.
            break; 
        } else {
            // If allocation was successful, print a success message.
            printf(" -> Successfully allocated %.2f GB of memory.\n", (double)limit / (1024*1024*1024));
            
            // Free the allocated memory before the next iteration.
            free(numbers);
        }
    }

    return 0;
}
