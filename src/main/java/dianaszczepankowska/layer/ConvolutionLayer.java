package dianaszczepankowska.layer;

import static dianaszczepankowska.tools.Matrix.add;
import static dianaszczepankowska.tools.Matrix.convolve;
import static dianaszczepankowska.tools.Matrix.flipHorizontal;
import static dianaszczepankowska.tools.Matrix.flipVertical;
import static dianaszczepankowska.tools.Matrix.fullConvolve;
import static dianaszczepankowska.tools.Matrix.multiply;
import static dianaszczepankowska.tools.Matrix.expandArray;
import static dianaszczepankowska.tools.Matrix.vectorToMatrix;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class ConvolutionLayer implements Layer {
    private Layer nextLayer;
    private Layer previousLayer;
    private List<double[][]> filters;
    private final int filterSize;
    private final int stepSize;
    private final int inputLength;
    private final int inputRows;
    private final int inputCols;
    private final long seed;
    private final double learningRate;
    private List<double[][]> lastInput;

    public ConvolutionLayer(int filterSize, int stepSize, int inputLength, int inputRows, int inputCols, int numberOfFilters, double learningRate, long seed) {
        this.filterSize = filterSize;
        this.stepSize = stepSize;
        this.inputLength = inputLength;
        this.inputRows = inputRows;
        this.inputCols = inputCols;
        this.seed = seed;
        this.learningRate = learningRate;
        generateRandomFilters(numberOfFilters);
    }

    @Override
    public double[] forwardPropagation(List<double[][]> input) {
        return nextLayer.forwardPropagation(forward(input));
    }

    @Override
    public double[] forwardPropagation(double[] input) {
        return forwardPropagation(vectorToMatrix(input, inputLength, inputRows, inputCols));
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPreviousLayer = new ArrayList<>();

        for (int f = 0; f < filters.size(); f++) {
            filtersDelta.add(new double[filterSize][filterSize]);
        }

        for (int i = 0; i < lastInput.size(); i++) {

            double[][] errorForInput = new double[inputRows][inputCols];

            for (int f = 0; f < filters.size(); f++) {
                double[][] currentFilter = filters.get(f);
                double[][] error = dLdO.get(i * filters.size() + f);

                double[][] spacedError = expandArray(error, stepSize);
                double[][] dLdF = convolve(lastInput.get(i), spacedError, 1);

                double[][] delta = multiply(dLdF, learningRate * -1);
                double[][] newTotalDelta = add(filtersDelta.get(f), delta);

                filtersDelta.set(f, newTotalDelta);

                double[][] flippedError = flipHorizontal(flipVertical(spacedError));
                errorForInput = add(errorForInput, fullConvolve(currentFilter, flippedError));

            }

            dLdOPreviousLayer.add(errorForInput);
        }
        for (int f = 0; f < filters.size(); f++) {
            double[][] modified = add(filtersDelta.get(f), filters.get(f));
            filters.set(f, modified);
        }

        if (previousLayer != null) {
            previousLayer.backPropagation(dLdOPreviousLayer);
        }

    }

    @Override
    public void backPropagation(double[] dLdO) {
        List<double[][]> matrixInput = vectorToMatrix(dLdO, inputLength, inputRows, inputCols);
        backPropagation(matrixInput);
    }

    @Override
    public int getOutputLength() {
        return filters.size() * inputLength;
    }

    @Override
    public int getOutputRows() {
        return (inputRows - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inputCols - filterSize) / stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return getOutputLength() * getOutputRows() * getOutputCols();
    }

    @Override
    public void setNextLayer(Layer layer) {
        this.nextLayer = layer;
    }

    @Override
    public void setPreviousLayer(Layer layer) {
        this.previousLayer = layer;
    }

    private List<double[][]> forward(List<double[][]> input) {
        lastInput = input;
        List<double[][]> output = new ArrayList<>();
        input.forEach(in -> filters.forEach(filter -> output.add(convolve(in, filter, stepSize))));
        return output;
    }

    private void generateRandomFilters(int numberOfFilters) {
        List<double[][]> filters = new ArrayList<>();
        Random random = new Random(seed);
        for (int f = 0; f < numberOfFilters; f++) {
            double[][] filter = new double[filterSize][filterSize];

            for (int i = 0; i < filterSize; i++) {
                for (int j = 0; j < filterSize; j++) {
                    filter[i][j] = random.nextGaussian();
                }
            }
            filters.add(filter);
        }
        this.filters = filters;
    }

}


