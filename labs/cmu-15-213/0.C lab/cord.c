/*
 * @file cord.c
 * @brief Implementation of cords library
 *
 * 15-513 Introduction to Computer Systems
 *
 * @author Foreverhighess <Foreverhighness@github.com>
 */

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "cord-interface.h"
#include "xalloc.h"

/***********************************/
/* Implementation (edit this part) */
/***********************************/

/**
 * @brief Checks if a cord is valid
 * @param[in] R
 * @return
 */
bool is_cord(const cord_t *R) {
    if (R == NULL)
        return true; /* is NULL */
    if (R->len == 0)
        return false; /* test circular */
    if (R->left == NULL && R->right == NULL)
        return R->len == strlen(R->data); /* is leaf */
    if (R->left != NULL && R->right != NULL) {
        return R->len == R->left->len + R->right->len && /* is non-leaf */
               is_cord(R->left) && is_cord(R->right);
    }
    return false;
}

/**
 * @brief Returns the length of a cord
 * @param[in] R
 * @return
 */
size_t cord_length(const cord_t *R) {
    if (R == NULL)
        return 0;
    return R->len;
}

/**
 * @brief Allocates a new cord from a string
 * @param[in] s
 * @return
 */
const cord_t *cord_new(const char *s) {
    if (s == NULL)
        return NULL;
    size_t len = strlen(s);
    if (len == 0)
        return NULL;
    cord_t *ret = malloc(sizeof(cord_t));
    ret->left = ret->right = NULL;
    ret->len = len;
    char *data = malloc(len + 1);
    strcpy(data, s);
    ret->data = data;
    return ret;
}

/**
 * @brief Concatenates two cords into a new cord
 * @param[in] R
 * @param[in] S
 * @return
 */
const cord_t *cord_join(const cord_t *R, const cord_t *S) {
    if (R == NULL)
        return S;
    if (S == NULL)
        return R;
    cord_t *ret = malloc(sizeof(cord_t));
    ret->left = R;
    ret->right = S;
    ret->len = R->len + S->len;
    ret->data = NULL;
    return ret;
}

/**
 * @brief Converts a cord to a string
 * @param[in] R
 * @return
 */
char *cord_tostring(const cord_t *R) {
    char *result = malloc(cord_length(R) + 1);
    *result = '\0';
    if (R == NULL)
        return result;                         /* is NULL */
    if (R->left == NULL && R->right == NULL) { /* is leaf */
        strcat(result, R->data);
        return result;
    }
    char *left = cord_tostring(R->left), *right = cord_tostring(R->right);
    strcat(result, left);  /* left part */
    strcat(result, right); /* right part */
    free(left);
    free(right);
    return result;
}

/**
 * @brief Returns the character at a position in a cord
 *
 * @param[in] R  A cord
 * @param[in] i  A position in the cord
 * @return The character at position i
 *
 * @requires i is a valid position in the cord R
 */
char cord_charat(const cord_t *R, size_t i) {
    assert(i <= cord_length(R));
    if (R->left == NULL && R->right == NULL) /* is leaf */
        return R->data[i];
    if (i < R->left->len)
        return cord_charat(R->left, i);
    return cord_charat(R->right, i - R->left->len);
}

/**
 * @brief Gets a substring of an existing cord
 *
 * @param[in] R   A cord
 * @param[in] lo  The low index of the substring, inclusive
 * @param[in] hi  The high index of the substring, exclusive
 * @return A cord representing the substring R[lo..hi-1]
 *
 * @requires lo and hi are valid indexes in R, with lo <= hi
 */
const cord_t *cord_sub(const cord_t *R, size_t lo, size_t hi) {
    assert(lo <= hi && hi <= cord_length(R));
    size_t len = hi - lo;
    if (len == 0)
        return NULL;
    if (len == R->len)
        return R;
    if (R->left == NULL && R->right == NULL) { /* is leaf */
        char *tmp = malloc(len + 1);
        memcpy(tmp, R->data + lo, len);
        tmp[len] = '\0';
        const cord_t *ret = cord_new(tmp);
        free(tmp);
        return ret;
    }

    if (hi <= R->left->len)
        return cord_sub(R->left, lo, hi);
    if (lo >= R->left->len)
        return cord_sub(R->right, lo - R->left->len, hi - R->left->len);

    return cord_join(cord_sub(R->left, lo, R->left->len),
                     cord_sub(R->right, 0, hi - R->left->len));
}

