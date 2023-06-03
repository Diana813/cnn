package dianaszczepankowska.layer;

import dianaszczepankowska.activationfunction.ActivationFunction;
import static dianaszczepankowska.tools.Matrix.matrixToVector;
import java.util.List;
import java.util.Random;

public class DenseLayer implements Layer {
    private Layer nextLayer;
    private Layer previousLayer;
    private final long seed;
    private final double[][] weights;
    private final int inputLength;
    private final int outputLength;
    private final double learningRate;
    private final ActivationFunction activationFunction;
    private double[] lastZ;
    private double[] lastX;

    public DenseLayer(int inputLength, int outputLength, double learningRate, ActivationFunction activationFunction, long seed) {
        this.inputLength = inputLength;
        this.outputLength = outputLength;
        this.seed = seed;
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;

        weights = new double[inputLength][outputLength];
        setRandomWeights();
    }


    @Override
    public double[] forwardPropagation(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return forwardPropagation(vector);
    }


    @Override
    public double[] forwardPropagation(double[] input) {
        double[] forwardPass = forwardPass(input);
        return nextLayer != null ? nextLayer.forwardPropagation(forwardPass) : forwardPass;
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        double[] vector = matrixToVector(dLdO);
        backPropagation(vector);
    }

    //dL/dW = dL/dO * dO/dZ * dZ/dW
    //dL/dX = dL/dO * dO/dZ * dZ/dX
    @Override
    public void backPropagation(double[] dLdO) {
        double[] dLdX = new double[inputLength];
        double dOdZ;
        double dZdW;
        double dLdW;
        double dZdX;


        for (int i = 0; i < inputLength; i++) {
            double dLdX_sum = 0;

            for (int j = 0; j < outputLength; j++) {
                dOdZ = activationFunction.derivative(lastZ[j]);
                dZdW = lastX[i];
                dLdW = dLdO[j] * dOdZ * dZdW;
                dZdX = weights[i][j];
                weights[i][j] -= dLdW * learningRate;
                dLdX_sum = dLdO[j] * dOdZ * dZdX;
            }

            dLdX[i] = dLdX_sum;
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return outputLength;
    }

    @Override
    public void setNextLayer(Layer layer) {
        this.nextLayer = layer;
    }

    @Override
    public void setPreviousLayer(Layer layer) {
        this.previousLayer = layer;
    }

    public void setRandomWeights() {
        Random random = new Random(seed);
        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                weights[i][j] = random.nextGaussian();
            }
        }
    }

    public double[] forwardPass(double[] input) {
        lastX = input;
        double[] z = new double[outputLength];
        double[] out = new double[outputLength];

        for (int j = 0; j < outputLength; j++) {
            for (int i = 0; i < inputLength; i++) {
                z[j] += input[i] * weights[i][j];
            }
            out[j] = activationFunction.apply(z[j]);
        }
        lastZ = z;
        return out;
    }
}
