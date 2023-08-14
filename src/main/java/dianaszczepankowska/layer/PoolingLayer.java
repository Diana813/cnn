package dianaszczepankowska.layer;

import static dianaszczepankowska.tools.Matrix.vectorToMatrix;
import java.util.ArrayList;
import java.util.List;

public class PoolingLayer implements Layer {
    private Layer nextLayer;
    private Layer previousLayer;
    private final int stepSize;
    private final int kernelSize;
    private final int inputLength;
    private final int inputRows;
    private final int inputCols;
    private final PoolingType type;
    private List<int[][]> lastPooledRows;
    private List<int[][]> lastPooledCols;

    public PoolingLayer(int stepSize, int kernelSize, int inputLength, int inputRows, int inputCols, PoolingType type) {
        this.stepSize = stepSize;
        this.kernelSize = kernelSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.type = type;
    }

    @Override
    public double[] forwardPropagation(List<double[][]> input) {
        List<double[][]> output = forwardPass(input);
        return nextLayer.forwardPropagation(output);
    }

    @Override
    public double[] forwardPropagation(double[] input) {
        List<double[][]> matrix = vectorToMatrix(input, inputLength, inputRows, inputCols);
        return nextLayer.forwardPropagation(matrix);
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        List<double[][]> dLdX = new ArrayList<>();
        int l = 0;

        for (double[][] array : dLdO) {
            double[][] error = new double[inputRows][inputCols];

            for (int r = 0; r < getOutputRows(); r += stepSize) {
                for (int c = 0; c < getOutputCols(); c += stepSize) {
                    int pooled_i = lastPooledRows.get(l)[r][c];
                    int pooled_j = lastPooledCols.get(l)[r][c];

                    if (pooled_i != -1) {
                        error[pooled_i][pooled_j] += array[r][c];
                    }
                }
            }
            dLdX.add(error);
            l++;
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(dLdX);
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrix = vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols());
        backPropagation(matrix);
    }

    @Override
    public int getOutputLength() {
        return inputLength;
    }

    @Override
    public int getOutputRows() {
        return (inputRows - kernelSize) / stepSize - 1;
    }

    @Override
    public int getOutputCols() {
        return (inputCols - kernelSize) / stepSize - 1;
    }

    @Override
    public int getOutputElements() {
        return inputLength * getOutputRows() * getOutputCols();
    }

    @Override
    public void setNextLayer(Layer layer) {
        this.nextLayer = layer;
    }

    @Override
    public void setPreviousLayer(Layer layer) {
        this.previousLayer = layer;
    }

    private List<double[][]> forwardPass(List<double[][]> input) {
        List<double[][]> output = new ArrayList<>();
        lastPooledCols = new ArrayList<>();
        lastPooledRows = new ArrayList<>();
        input.forEach(in -> output.add(type.applyPooling(kernelSize, stepSize, in, getOutputRows(), getOutputCols(), lastPooledRows, lastPooledCols)));
        return output;
    }

    @Override
    public String toString() {
        return "PoolingLayer{" +
                "filterSize: " + kernelSize +
                ", stepSize: " + stepSize +
                ", pooling type: " + type.name() +
                '}';
    }

}
