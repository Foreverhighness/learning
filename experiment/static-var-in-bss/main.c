#include <stdio.h>

static int S_VAR_1;
static int S_VAR_2 = 2;
const int C_VAR_3 = 3;
static const int S_VAR_4 = 4;
const int C_VAR_5;

void foo() {
    static int S_VAR_6;
    static int S_VAR_7 = 7;
    const int C_VAR_8 = 8;
    static const int C_VAR_9 = 9;
    static const int C_VAR_10;
    printf("%d\n", S_VAR_1);
    printf("%d\n", S_VAR_2);
    printf("%d\n", C_VAR_3);
    printf("%d\n", S_VAR_4);
    printf("%d\n", C_VAR_5);
    printf("%d\n", S_VAR_6);
    printf("%d\n", S_VAR_7);
    printf("%d\n", C_VAR_8);
    printf("%d\n", C_VAR_9);
    printf("%d\n", C_VAR_10);
    S_VAR_1 = 11;
    S_VAR_2 = 22;
    S_VAR_6 = 66;
    S_VAR_7 = 77;
    printf("%d\n", S_VAR_1);
    printf("%d\n", S_VAR_2);
    printf("%d\n", C_VAR_3);
    printf("%d\n", S_VAR_4);
    printf("%d\n", C_VAR_5);
    printf("%d\n", S_VAR_6);
    printf("%d\n", S_VAR_7);
    printf("%d\n", C_VAR_8);
    printf("%d\n", C_VAR_9);
    printf("%d\n", C_VAR_10);
}

int main() {
    foo();
    return 0;
}

