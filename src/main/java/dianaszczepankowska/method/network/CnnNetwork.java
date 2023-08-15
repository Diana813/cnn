package dianaszczepankowska.method.network;

import dianaszczepankowska.problem.model.Image;
import dianaszczepankowska.method.layer.Layer;
import static dianaszczepankowska.method.tools.Matrix.add;
import static dianaszczepankowska.method.tools.Matrix.multiply;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

public class CnnNetwork {
    List<Layer> layers;

    public CnnNetwork(List<Layer> layers) {
        this.layers = layers;
        connectLayers();
    }

    public double test(List<Image> images) {
        long correct = images.stream()
                .filter(img -> classifyImage(img) == img.label())
                .count();

        return (double) correct / images.size();
    }

    public void train(List<Image> images) {
        images.forEach(image -> {
            List<double[][]> inputList = new ArrayList<>();
            inputList.add(image.pixels());

            double[] out = layers.get(0).forwardPropagation(inputList);
            double[] dldO = calculateErrorVector(out, image.label());

            layers.get(layers.size() - 1).backPropagation(dldO);
        });
    }

    public double[] predictProbabilities(Image images) {
        return layers.get(0).forwardPropagation(Collections.singletonList(images.pixels()));
    }

    private int classifyImage(Image image) {
        List<double[][]> inputList = new ArrayList<>();
        inputList.add(image.pixels());

        double[] out = layers.get(0).forwardPropagation(inputList);

        return findIndexOfMaxValue(out);
    }

    private void connectLayers() {
        if (layers.size() <= 1) {
            return;
        }

        for (int i = 0; i < layers.size(); i++) {
            if (i == 0) {
                layers.get(i).setNextLayer(layers.get(i + 1));
            } else if (i == layers.size() - 1) {
                layers.get(i).setPreviousLayer(layers.get(i - 1));
            } else {
                layers.get(i).setPreviousLayer(layers.get(i - 1));
                layers.get(i).setNextLayer(layers.get(i + 1));
            }
        }
    }

    private double[] calculateErrorVector(double[] networkOutput, int correctAnswer) {
        int labelsNumber = networkOutput.length;

        double[] expected = new double[labelsNumber];

        expected[correctAnswer] = 1;

        return add(networkOutput, multiply(expected, -1));
    }

    private int findIndexOfMaxValue(double[] in) {
        return IntStream.range(0, in.length)
                .reduce((i, j) -> in[i] >= in[j] ? i : j)
                .orElse(0);
    }

}
