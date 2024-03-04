#include "headers/includes.h"
#include "headers/sigmoid.h"

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1 - x);
}