package dianaszczepankowska.method.layer;

import dianaszczepankowska.method.activationfunction.ActivationFunction;
import dianaszczepankowska.method.activationfunction.SoftMax;
import dianaszczepankowska.method.tools.Matrix;
import java.util.Collections;
import java.util.List;

public class SoftmaxLayer implements Layer{

    private Layer nextLayer;
    private Layer previousLayer;
    private final int outputLength;
    private double[] output;

    private final ActivationFunction activationFunction;

    public SoftmaxLayer(int outputLength){
        this.outputLength = outputLength;
        this.activationFunction = new SoftMax();
    }
    @Override
    public double[] forwardPropagation(List<double[][]> input) {
        return new double[0];
    }

    @Override
    public double[] forwardPropagation(double[] input) {
        output = activationFunction.apply(input);
        return nextLayer != null ? nextLayer.forwardPropagation(output) : output;
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

    }

    @Override
    public void backPropagation(double[] dLdO) {
        int n = output.length;
        double[][] outputMatrix = Matrix.tileArray(output, n);
        double[][] dLdX = Matrix.dotProduct(Matrix.multiply(outputMatrix, Matrix.subtract(Matrix.identity(n), Matrix.transpose(outputMatrix))), Matrix.arrayToColumnMatrix(dLdO));

        if (previousLayer != null) {
            previousLayer.backPropagation(Collections.singletonList(dLdX));
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

    @Override
    public String toString() {
        return super.toString();
    }
}
