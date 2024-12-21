#include <stdio.h>
#include <stdlib.h>
#include <math.h>
double* y_(double w, double b, double* x, int N);

int main(void)
{
    int N = 0; // 训练样本量
    scanf("%d", &N); // 读取训练样本数量
    double* x = calloc(N, sizeof(double));
    double* y = calloc(N, sizeof(double));

    // 读取训练样本数据
    for (int cnt = 0; cnt < N; cnt++) {
        scanf("%lf %lf", &x[cnt], &y[cnt]);
    }

    // 初始化权重，偏置，训练次数，学习率和观测值
    double w = 0, b = 0;
    int epoch = 10000;
    double alpha = 0.01;
    double* y_bar = y_(w, b, x, N);


    // 开始训练
    for (int cnt = 0; cnt < epoch; cnt++) {
        double delta_w = 0;
        double delta_b = 0;
        y_bar = y_(w, b, x, N);
        for (int i = 0; i < N; i++) {
            delta_w += (y_bar[i] - y[i]) * x[i];
            delta_b += (y_bar[i] - y[i]);
        }
        delta_w = -1.0 * delta_w / N * alpha;
        delta_b = -1.0 * delta_b / N * alpha;
        w = w + delta_w;
        b = b + delta_b;
        printf("Epoch %d: w = %.16lf, b = %.16lf\n", cnt, w, b);
    }

}

double* y_(double w, double b, double* x, int N)
{
    double* ret = calloc(N, sizeof(double));
    for (int cnt = 0; cnt < N; cnt++) {
        ret[cnt] = w * x[cnt] + b;
    }
    return ret;
}