#include <stdio.h>

int add(int a, int b);

int main() {
    int a, b;
    printf("input two number:\n");
    while ((scanf("%d %d", &a, &b) == 2)) {
        printf("%d + %d = %d\n", a, b, add(a, b));
        printf("input two number:\n");
    }
    return 0;
}
