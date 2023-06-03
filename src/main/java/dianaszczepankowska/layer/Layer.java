package dianaszczepankowska.layer;

import java.util.List;

public interface Layer {

    double[] forwardPropagation(List<double[][]> input);

    double[] forwardPropagation(double[] input);

    void backPropagation(List<double[][]> dLdO);

    void backPropagation(double[] dLdO);

    int getOutputLength();

    int getOutputRows();

    int getOutputCols();

    int getOutputElements();

    void setNextLayer(Layer layer);

    void setPreviousLayer(Layer layer);
}
