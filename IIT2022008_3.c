/*
Parallel and Distributed Computing Assignment 2, done by Shashank Shashidhar IIT2022008

Counting prime numbers from 1 to 2^n where n is the maximum possible n on my system (n = 32)
Experimental Setup:
CPU architecture: x86_64
CPU op-mode(s):         32-bit, 64-bit
Address sizes:          39 bits physical, 48 bits virtual
CPU(s): 8 with 4 cores and 2 threads per core. 
Compilation of code is done on WSL, using gcc compiler with -pthread 

Results:
for 2 threads ->
Sequential execution time: 0.000003 seconds
Parallel execution time:   0.001213 seconds

for 4 threads->
Sequential execution time: 0.000009 seconds
Parallel execution time:   0.002013 seconds

for 8 threads ->
Sequential execution time: 0.000004 seconds
Parallel execution time:   0.003274 seconds

for 16 threads ->
Sequential execution time: 0.000009 seconds
Parallel execution time:   0.012834 seconds

for 32 threads -> 
Sequential execution time: 0.000004 seconds
Parallel execution time:   0.012350 seconds

Using more threads increases the amount of time required to execute the sparse matrix vector multiplication. This is due
to the fact that the input matrix is only a 138x138 diagonal matrix. The number of computations that is required for 
such a small matrix is not too big and is supported very easily by modern CPUs. Each thread doesnt have too much work to 
do. There is more overhead in creating and managing the thread than there is in actually executing the process.

HOW THE CODE WORKS:
The code first takes in the matrix, the vector and the number of threads as command line args. The COO representation is first
converted to a CSR representation. The multiplcation algorithm is used for matrix product with the vector. The algorithm
ensures that multiplication is only done between the non zero elements of the matrix and the vector. This is made possible
by the CSR representation, which gives us an efficient representation of which positions in the matrix contains non zero elements.
C1 and C2 are seen to be the same in both sequential and parallel executions. The time is recorded and printed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h> 

typedef struct {
    int num_rows;
    int num_cols;
    int num_non_zeros;
    double *values;
    int *col_indices;
    int *row_pointers;
} SparseMatrixCSR;

typedef struct {
    int thread_id;
    int num_threads;
    const SparseMatrixCSR *A;
    const double *B;
    double *C;
} ThreadData;

SparseMatrixCSR read_and_convert_to_csr(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening matrix file");
        exit(EXIT_FAILURE);
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (line[0] != '%') {
            break;
        }
    }

    int num_rows, num_cols, num_non_zeros;
    sscanf(line, "%d %d %d", &num_rows, &num_cols, &num_non_zeros);

    SparseMatrixCSR A;
    A.num_rows = num_rows;
    A.num_cols = num_cols;
    A.num_non_zeros = num_non_zeros;
    A.values = (double *)malloc(num_non_zeros * sizeof(double));
    A.col_indices = (int *)malloc(num_non_zeros * sizeof(int));
    A.row_pointers = (int *)calloc(num_rows + 1, sizeof(int));

    if (!A.values || !A.col_indices || !A.row_pointers) {
        fprintf(stderr, "Memory allocation failed for CSR matrix.\n");
        exit(EXIT_FAILURE);
    }

    int *coo_rows = (int *)malloc(num_non_zeros * sizeof(int));
    int *coo_cols = (int *)malloc(num_non_zeros * sizeof(int));
    double *coo_vals = (double *)malloc(num_non_zeros * sizeof(double));
    if (!coo_rows || !coo_cols || !coo_vals) {
        fprintf(stderr, "Memory allocation failed for temporary COO arrays.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < num_non_zeros; ++i) {
        int row, col;
        double val;
        if (fscanf(file, "%d %d %lf", &row, &col, &val) != 3) {
            fprintf(stderr, "Error reading matrix data.\n");
            exit(EXIT_FAILURE);
        }
        coo_rows[i] = row - 1;
        coo_cols[i] = col - 1;
        coo_vals[i] = val;
        A.row_pointers[row]++;
    }

    fclose(file);

    for (int i = 0; i < num_rows; ++i) {
        A.row_pointers[i + 1] += A.row_pointers[i];
    }
    
    int *row_counts = (int *)calloc(num_rows, sizeof(int));
    if (!row_counts) {
        fprintf(stderr, "Memory allocation failed for row_counts.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_non_zeros; ++i) {
        int row = coo_rows[i];
        int col = coo_cols[i];
        double val = coo_vals[i];

        int pos = A.row_pointers[row] + row_counts[row];
        A.values[pos] = val;
        A.col_indices[pos] = col;
        row_counts[row]++;
    }

    free(coo_rows);
    free(coo_cols);
    free(coo_vals);
    free(row_counts);

    return A;
}

double* read_vector(const char* filename, int *vector_size) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening vector file");
        exit(EXIT_FAILURE);
    }
    
    double* vec = NULL;
    int capacity = 100; 
    vec = (double*)malloc(capacity * sizeof(double));
    if (!vec) {
        fprintf(stderr, "Memory allocation failed for vector.\n");
        exit(EXIT_FAILURE);
    }
    *vector_size = 0;

    char line[4096];
    while(fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",\r\n");
        while(token != NULL) {
            if (*vector_size >= capacity) {
                capacity *= 2;
                vec = (double*)realloc(vec, capacity * sizeof(double));
                if (!vec) {
                    fprintf(stderr, "Memory reallocation failed.\n");
                    exit(EXIT_FAILURE);
                }
            }
            vec[*vector_size] = strtod(token, NULL);
            (*vector_size)++;
            token = strtok(NULL, ",\r\n");
        }
    }

    fclose(file);
    return vec;
}

void multiply_sequential(const SparseMatrixCSR *A, const double *B, double *C) {
    for (int i = 0; i < A->num_rows; ++i) {
        C[i] = 0.0;
        int start_idx = A->row_pointers[i];
        int end_idx = A->row_pointers[i + 1];
        for (int j = start_idx; j < end_idx; ++j) {
            C[i] += A->values[j] * B[A->col_indices[j]];
        }
    }
}

void* multiply_parallel_thread_func(void* arg) {
    ThreadData *data = (ThreadData *)arg; //typecast to thread data

    int start_row = data->thread_id * (data->A->num_rows / data->num_threads);
    int end_row = start_row + (data->A->num_rows / data->num_threads);
    if (data->thread_id == data->num_threads - 1) {
        end_row = data->A->num_rows;
    }

    for (int i = start_row; i < end_row; ++i) {
        data->C[i] = 0.0;
        int start_idx = data->A->row_pointers[i];
        int end_idx = data->A->row_pointers[i + 1];
        for (int j = start_idx; j < end_idx; ++j) {
            data->C[i] += data->A->values[j] * data->B[data->A->col_indices[j]];
        }
    }

    pthread_exit(NULL);
}

void print_matrix_csr(const SparseMatrixCSR *A, int num_threads) {
    printf("\n#Rows: %d\n", A->num_rows);
    printf("#Cols: %d\n", A->num_cols);
    printf("#Non-Zeroes: %d\n", A->num_non_zeros);
    printf("#threads: %d\n\n", num_threads);

    printf("Matrix (CSR format):\n");
    printf("Values: ");
    for (int i = 0; i < A->num_non_zeros; ++i) {
        printf("%.2f ", A->values[i]);
    }
    printf("\n");

    printf("Column Indices: ");
    for (int i = 0; i < A->num_non_zeros; ++i) {
        printf("%d ", A->col_indices[i]);
    }
    printf("\n");

    printf("Row Pointers: ");
    for (int i = 0; i <= A->num_rows; ++i) {
        printf("%d ", A->row_pointers[i]);
    }
    printf("\n\n");
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx> <vector_file.txt> <num_threads>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *matrix_filename = argv[1];
    const char *vector_filename = argv[2];
    int num_threads = atoi(argv[3]);

    if (num_threads <= 0) {
        fprintf(stderr, "Number of threads must be a positive integer.\n");
        exit(EXIT_FAILURE);
    }

    SparseMatrixCSR A = read_and_convert_to_csr(matrix_filename);
    
    int vector_size;
    double* B = read_vector(vector_filename, &vector_size);

    print_matrix_csr(&A, num_threads);

    //sequential computation
    double* C1 = (double*)calloc(A.num_rows, sizeof(double));
    if (!C1) {
        fprintf(stderr, "Memory allocation failed for C1.\n");
        exit(EXIT_FAILURE);
    }
    struct timespec start_seq, end_seq;
    clock_gettime(CLOCK_MONOTONIC, &start_seq);
    multiply_sequential(&A, B, C1);
    clock_gettime(CLOCK_MONOTONIC, &end_seq);
    double seq_time = (end_seq.tv_sec - start_seq.tv_sec) + (end_seq.tv_nsec - start_seq.tv_nsec) / 1e9;

    printf("Sequential Result (C1):\n");
    for (int i = 0; i < A.num_rows; ++i) {
        printf("%.4f ", C1[i]);
    }
    printf("\n\n");

    //parallel computation
    double* C2 = (double*)calloc(A.num_rows, sizeof(double));
    if (!C2) {
        fprintf(stderr, "Memory allocation failed for C2.\n");
        exit(EXIT_FAILURE);
    }
    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t)); //initialize threads
    ThreadData *thread_data_array = (ThreadData*)malloc(num_threads * sizeof(ThreadData)); //initialize thread data
    if (!threads || !thread_data_array) {
        fprintf(stderr, "Memory allocation failed for threads or thread data.\n");
        exit(EXIT_FAILURE);
    }

    struct timespec start_par, end_par;
    clock_gettime(CLOCK_MONOTONIC, &start_par);
    for (int i = 0; i < num_threads; ++i) {
        thread_data_array[i].thread_id = i;
        thread_data_array[i].num_threads = num_threads;
        thread_data_array[i].A = &A;
        thread_data_array[i].B = B;
        thread_data_array[i].C = C2;
        pthread_create(&threads[i], NULL, multiply_parallel_thread_func, &thread_data_array[i]);
    }

    //join threads
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_par);
    double par_time = (end_par.tv_sec - start_par.tv_sec) + (end_par.tv_nsec - start_par.tv_nsec) / 1e9;

    printf("Parallel Result (C2):\n");
    for (int i = 0; i < A.num_rows; ++i) {
        printf("%.4f ", C2[i]);
    }
    printf("\n\n");
    
    printf("Sequential execution time: %lf seconds\n", seq_time);
    printf("Parallel execution time:   %lf seconds\n", par_time);

    free(A.values);
    free(A.col_indices);
    free(A.row_pointers);
    free(B);
    free(C1);
    free(C2);
    free(threads);
    free(thread_data_array);

    return 0;
}
