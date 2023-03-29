#include "derive.h"

static volatile int dummy;

void main_init() {
    dummy = 3;
}

void main_print() {
    printf("%d\n", dummy);
}

int main() {
    common_init();
    derive_init();
    main_init();

    common_print();
    derive_print();
    main_print();

    return 0;
}
