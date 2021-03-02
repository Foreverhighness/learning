/*
 * trans.size - Matrix transpose B = A^T
 *
 * Each transpose function must have a prototype of the form:
 * void trans(int M, int N, int A[N][M], int B[M][N]);
 *
 * A transpose function is evaluated by counting the number of misses
 * on a 1KB direct mapped cache with a block size of 32 bytes.
 */
#include <stdio.h>
#include <assert.h>
#include "cachelab.h"

int is_transpose(int M, int N, int A[N][M], int B[M][N]);

/*
 * transpose_submit - This is the solution transpose function that you
 *     will be graded on for Part B of the assignment. Do not change
 *     the description string "Transpose submission", as the driver
 *     searches for that string to identify the transpose function to
 *     be graded.
 */
char transpose_submit_desc[] = "Transpose submission";
void transpose_submit(int M, int N, int A[N][M], int B[M][N])
{
    if (N == 32) {
        const int size = 8;
        int ii, jj, i;
        int t0, t1, t2, t3, t4, t5, t6, t7;
        for (ii = 0; ii < N; ii += size) {
            for (jj = 0; jj < M; jj += size) {
                for (i = ii; i < ii + size; ++i) {
                    t0 = A[i][jj + 0];
                    t1 = A[i][jj + 1];
                    t2 = A[i][jj + 2];
                    t3 = A[i][jj + 3];
                    t4 = A[i][jj + 4];
                    t5 = A[i][jj + 5];
                    t6 = A[i][jj + 6];
                    t7 = A[i][jj + 7];

                    B[jj + 0][i] = t0;
                    B[jj + 1][i] = t1;
                    B[jj + 2][i] = t2;
                    B[jj + 3][i] = t3;
                    B[jj + 4][i] = t4;
                    B[jj + 5][i] = t5;
                    B[jj + 6][i] = t6;
                    B[jj + 7][i] = t7;
                }
            }
        }
    } else if (N == 64) {
        const int size = 8;
        const int half_size = 4;
        int ii, jj, i, j;
        int t0, t1, t2, t3, t4, t5, t6, t7;
        for (ii = 0; ii < N; ii += size) {
            for (jj = 0; jj < M; jj += size) {
                for (i = ii; i < ii + half_size; ++i) {
                    t0 = A[i][jj + 0];
                    t1 = A[i][jj + 1];
                    t2 = A[i][jj + 2];
                    t3 = A[i][jj + 3];
                    t4 = A[i][jj + 4];
                    t5 = A[i][jj + 5];
                    t6 = A[i][jj + 6];
                    t7 = A[i][jj + 7];

                    B[jj + 0][i] = t0;
                    B[jj + 1][i] = t1;
                    B[jj + 2][i] = t2;
                    B[jj + 3][i] = t3;

                    B[jj + 0][i + half_size] = t4;
                    B[jj + 1][i + half_size] = t5;
                    B[jj + 2][i + half_size] = t6;
                    B[jj + 3][i + half_size] = t7;
                }
                for (j = jj; j < jj + half_size; ++j) {
                    t0 = B[j][ii + half_size + 0];
                    t1 = B[j][ii + half_size + 1];
                    t2 = B[j][ii + half_size + 2];
                    t3 = B[j][ii + half_size + 3];

                    B[j][ii + half_size + 0] = A[ii + half_size + 0][j];
                    B[j][ii + half_size + 1] = A[ii + half_size + 1][j];
                    B[j][ii + half_size + 2] = A[ii + half_size + 2][j];
                    B[j][ii + half_size + 3] = A[ii + half_size + 3][j];

                    B[j + half_size][ii + 0] = t0;
                    B[j + half_size][ii + 1] = t1;
                    B[j + half_size][ii + 2] = t2;
                    B[j + half_size][ii + 3] = t3;
                    B[j + half_size][ii + 4] = A[ii + half_size + 0][j + half_size];
                    B[j + half_size][ii + 5] = A[ii + half_size + 1][j + half_size];
                    B[j + half_size][ii + 6] = A[ii + half_size + 2][j + half_size];
                    B[j + half_size][ii + 7] = A[ii + half_size + 3][j + half_size];
                }
            }
        }
    } else {
        const int size = 16;
        int ii, jj, i, j;
        for (ii = 0; ii < N; ii += size) {
            for (jj = 0; jj < M; jj += size) {
                for (i = ii; i < ii + size && i < N; ++i) {
                    for (j = jj; j < jj + size && j < M; ++j) {
                        B[j][i] = A[i][j];
                    }
                }
            }
        }
    }
}

/*
 * You can define additional transpose functions below. We've defined
 * a simple one below to help you get started.
 */

/*
 * trans - A simple baseline transpose function, not optimized for the cache.
 */
char trans_desc[] = "Simple row-wise scan transpose";
void trans(int M, int N, int A[N][M], int B[M][N])
{
    int i, j, tmp;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            tmp = A[i][j];
            B[j][i] = tmp;
        }
    }

}

/*
 * registerFunctions - This function registers your transpose
 *     functions with the driver.  At runtime, the driver will
 *     evaluate each of the registered functions and summarize their
 *     performance. This is a handy way to experiment with different
 *     transpose strategies.
 */
void registerFunctions()
{
    /* Register your solution function */
    registerTransFunction(transpose_submit, transpose_submit_desc);

    /* Register any additional transpose functions */
    // registerTransFunction(trans, trans_desc);

}

/*
 * is_transpose - This helper function checks if B is the transpose of
 *     A. You can check the correctness of your transpose by calling
 *     it before returning from the transpose function.
 */
int is_transpose(int M, int N, int A[N][M], int B[M][N])
{
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; ++j) {
            if (A[i][j] != B[j][i]) {
                return 0;
            }
        }
    }
    return 1;
}

