#include <stdio.h>
#include <math.h>
#include <stdlib.h>
double np_dot(double* w, double* x, double N);
double dj_dw(double* w, double** x, double* y, double b, int M, int N, int j);
double dj_db(double* w, double** x, double* y, double b, int M, int N);
double square_loss(double* w, double** x, double b, double* y, int M, int N);

int main(void)
{
    int M = 0, N = 0; // M：训练样本批次数 N：单批样本特征数
    scanf("%d %d", &M, &N);
    double** X_train = calloc(M, sizeof(double*));
    for (int cnt = 0; cnt < M; cnt++) {
        X_train[cnt] = calloc(N, sizeof(double));
    }
    double* y_train = calloc(M, sizeof(double));

    // 初始化训练样本
    for (int i = 0; i < M; i++) {
        printf("%d batch of X_train: ", i + 1);
        for (int j = 0; j < N; j++) {
            scanf("%lf", &X_train[i][j]);
        }
        printf("%d batch of y_train: ", i + 1);
        scanf("%lf", &y_train[i]);
    }

    // 初始化权重，偏置等训练参数
    double* w = calloc(N, sizeof(double));
    double b = 0;
    double alpha = 0.01;
    int batch_size = 7000;

    // 开始训练
    for (int cnt = 0; cnt < batch_size; cnt++) {
        double* delta_w = calloc(N, sizeof(double));
        double delta_b = 0;
        for (int j = 0; j < N; j++) {
            delta_w[j] = (-1) * alpha * dj_dw(w, X_train, y_train, b, M, N, j);
            w[j] = w[j] + delta_w[j];
        }
        delta_b = (-1) * alpha * dj_db(w, X_train, y_train, b, M, N);
        b += delta_b;
        double loss = square_loss(w, X_train, b, y_train, M, N);
        printf("Epoch %d: loss = %.16lf\n", cnt + 1, loss);
        free(delta_w);
    }
    printf("\nFinal result: \n");
    for (int j = 0; j < N; j++) {
        printf("W_%d=%.6lf, ", j, w[j]);
    }
    printf("\nb=%.6lf", b);

    // 释放内存
    for (int cnt = 0; cnt < M; cnt++) {
        free(X_train[cnt]);
    }
    free(X_train);
    free(y_train);
    free(w);

    return 0;
}

double np_dot(double* w, double* x, double N)
{
    double ret = 0;
    for (int cnt = 0; cnt < N; cnt++) {
        ret += w[cnt] * x[cnt];
    }
    return ret;
}

double dj_dw(double* w, double** x, double* y, double b, int M, int N, int j)
{
    double ret = 0;
    for (int i = 0; i < M; i++) {
        ret += (np_dot(w, x[i], N) + b - y[i]) * x[i][j];
    }
    ret /= M;
    return ret;
}

double dj_db(double* w, double** x, double* y, double b, int M, int N)
{
    double ret = 0;
    for (int i = 0; i < M; i++) {
        ret += np_dot(w, x[i], N) + b - y[i];
    }
    return ret;
}

double square_loss(double* w, double** x, double b, double* y, int M, int N)
{
    double ret = 0;
    for (int i = 0; i < M; i++) {
        double prediction = np_dot(w, x[i], N) + b;
        ret += pow((prediction - y[i]), 2);
    }
    ret /= (2 * M);
    return ret;
}