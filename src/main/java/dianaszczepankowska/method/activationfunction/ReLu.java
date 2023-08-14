package dianaszczepankowska.method.activationfunction;

public class ReLu implements ActivationFunction {
    @Override
    public double apply(double input) {
        return input <= 0 ? 0 : input;
    }

    @Override
    public double derivative(double input) {
        double leak = 0.01;
        return input <= 0 ? leak : 1;
    }
}
