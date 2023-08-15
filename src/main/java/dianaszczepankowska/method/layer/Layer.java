package dianaszczepankowska.method.layer;

import java.util.List;

public interface Layer {

    double[] forwardPropagation(List<double[][]> input);

    double[] forwardPropagation(double[] input);

    void backPropagation(List<double[][]> dLdO);

    void backPropagation(double[] dLdO);

    int getOutputElements();

    void setNextLayer(Layer layer);

    void setPreviousLayer(Layer layer);

    default int getOutputLength() {
        return 0;
    }

    default int getOutputRows() {
        return 0;
    }

    default int getOutputCols() {
        return 0;
    }

}
