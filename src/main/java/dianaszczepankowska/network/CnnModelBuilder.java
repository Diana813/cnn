package dianaszczepankowska.network;

import dianaszczepankowska.activationfunction.ActivationFunction;
import dianaszczepankowska.layer.ConvolutionLayer;
import dianaszczepankowska.layer.DenseLayer;
import dianaszczepankowska.layer.Layer;
import dianaszczepankowska.layer.PoolingLayer;
import dianaszczepankowska.layer.PoolingType;
import java.util.ArrayList;
import java.util.List;

public class CnnModelBuilder {

    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;
    private final List<Layer> layers = new ArrayList<>();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }


    public CnnModelBuilder addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long seed) {
        if (layers.isEmpty()) {
            addLayer(new ConvolutionLayer(filterSize, stepSize, 1, IMAGE_HEIGHT, IMAGE_WIDTH, numFilters, learningRate, seed));
        } else {
            Layer previous = layers.get(layers.size() - 1);
            addLayer(new ConvolutionLayer(filterSize, stepSize, previous.getOutputLength(), previous.getOutputRows(), previous.getOutputCols(), numFilters, learningRate, seed));
        }
        return this;
    }

    public CnnModelBuilder addPoolingLayer(int kernelSize, int stepSize, PoolingType type) {
        if (layers.isEmpty()) {
            addLayer(new PoolingLayer(stepSize, kernelSize, 1, IMAGE_HEIGHT, IMAGE_WIDTH, type));
        } else {
            Layer previous = layers.get(layers.size() - 1);
            addLayer(new PoolingLayer(stepSize, kernelSize, previous.getOutputLength(), previous.getOutputRows(), previous.getOutputCols(), type));
        }
        return this;
    }

    public CnnModelBuilder addDenseLayer(int outLength, double learningRate, ActivationFunction activationFunction, long seed) {
        if (layers.isEmpty()) {
            addLayer(new DenseLayer(IMAGE_HEIGHT * IMAGE_WIDTH, outLength, learningRate, activationFunction, seed));
        } else {
            Layer previous = layers.get(layers.size() - 1);
            addLayer(new DenseLayer(previous.getOutputElements(), outLength, learningRate, activationFunction, seed));
        }
        return this;
    }

    public CnnNetwork build() {
        return new CnnNetwork(layers);
    }

}