package dianaszczepankowska.method.network;

import dianaszczepankowska.method.activationfunction.ActivationFunction;
import dianaszczepankowska.method.layer.ConvolutionLayer;
import dianaszczepankowska.method.layer.DenseLayer;
import dianaszczepankowska.method.layer.Layer;
import dianaszczepankowska.method.layer.PoolingLayer;
import dianaszczepankowska.method.layer.PoolingType;
import dianaszczepankowska.method.layer.SoftmaxLayer;
import java.util.ArrayList;
import java.util.List;

public class CnnModelBuilder {

    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;
    private final List<Layer> layers = new ArrayList<>();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public CnnModelBuilder addConvolutionLayer(int numFilters, int filterSize, int stepSize,
                                               double learningRate, boolean padding, long seed) {
        Layer previous = getPreviousLayer();
        int outputLength = previous == null ? 1 : previous.getOutputLength();
        int outputRows = previous == null ? IMAGE_HEIGHT : previous.getOutputRows();
        int outputCols = previous == null ? IMAGE_WIDTH : previous.getOutputCols();

        addLayer(new ConvolutionLayer(filterSize, stepSize, outputLength, outputRows, outputCols,
                numFilters, learningRate, seed, padding));

        return this;
    }

    public CnnModelBuilder addPoolingLayer(int kernelSize, int stepSize, PoolingType type) {
        Layer previous = getPreviousLayer();
        int outputLength = previous == null ? 1 : previous.getOutputLength();
        int outputRows = previous == null ? IMAGE_HEIGHT : previous.getOutputRows();
        int outputCols = previous == null ? IMAGE_WIDTH : previous.getOutputCols();

        addLayer(new PoolingLayer(stepSize, kernelSize, outputLength, outputRows, outputCols, type));

        return this;
    }

    public CnnModelBuilder addDenseLayer(int outLength, double learningRate,
                                         ActivationFunction activationFunction, long seed) {
        Layer previous = getPreviousLayer();
        int outputElements = previous == null ? IMAGE_HEIGHT * IMAGE_WIDTH : previous.getOutputElements();

        addLayer(new DenseLayer(outputElements, outLength, learningRate, activationFunction, seed));

        return this;
    }

    public CnnModelBuilder addSoftMaxLayer(int outLength) {
        addLayer(new SoftmaxLayer(outLength));

        return this;
    }

    public CnnNetwork build() {
        return new CnnNetwork(layers);
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        for (Layer layer : layers) {
            builder.append(layer.toString()).append("\n");
        }
        return builder.toString();
    }

    private Layer getPreviousLayer() {
        if (layers.isEmpty()) {
            return null;
        }
        return layers.get(layers.size() - 1);
    }
}