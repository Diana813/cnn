package dianaszczepankowska.method.activationfunction;

public class SoftMax implements ActivationFunction {
    @Override
    public double apply(double x) {
        return 0;
    }

    @Override
    public double derivative(double x) {
        return 0;
    }

    @Override
    public double[] apply(double[] x) {
        double[] result = new double[x.length];
        double sumExp = 0.0;

        for (double value : x) {
            sumExp += Math.exp(value);
        }

        for (int i = 0; i < x.length; i++) {
            result[i] = Math.exp(x[i]) / sumExp;
        }

        return result;
    }

    @Override
    public double[] derivative(double[] x) {
        int n = x.length;
        double[] derivative = new double[n];
        double[] softmax = apply(x);

        for (int i = 0; i < n; i++) {
            derivative[i] = softmax[i] * (1 - softmax[i]);
        }

        return derivative;
    }


}
