#if !defined(DATASETS_H)
#define DATASETS_H

#include "nn.h"

void getXorNum(int* trainingSet, int* inputsLength, int* targetsLength);
NeuralNetwork* getXorDataset(double** inputs, double** targets, double* predictInput, double* predictTarget);
void getDataOneOutputNum(int* trainingSet, int* inputsLength, int* targetsLength);
NeuralNetwork* getDataOneOutputDataset(double** inputs, double** targets, double* predictInput, double* predictTarget);
void getDataTwoOutputNum(int* trainingSet, int* inputsLength, int* targetsLength);
NeuralNetwork* getDataTwoOutputsDataset(double** inputs, double** targets, double* predictInput, double* predictTarget);

#endif