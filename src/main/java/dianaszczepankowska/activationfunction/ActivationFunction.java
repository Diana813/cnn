package dianaszczepankowska.activationfunction;


public interface ActivationFunction {
    double apply(double x);
    double derivative(double x);

    default double[] apply(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = apply(x[i]);
        }
        return result;
    }

    default double[] derivative(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = derivative(x[i]);
        }
        return result;
    }

    default double[][] apply(double[][] x) {
        double[][] result = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            result[i] = apply(x[i]);
        }
        return result;
    }

    default double[][] derivative(double[][] x) {
        double[][] result = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            result[i] = derivative(x[i]);
        }
        return result;
    }
}
