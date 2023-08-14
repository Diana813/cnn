package dianaszczepankowska.method.layer;

import static dianaszczepankowska.method.tools.Matrix.averagePooling;
import static dianaszczepankowska.method.tools.Matrix.maxPooling;
import java.util.List;

public enum PoolingType {
    MAX {
        @Override
        public double[][] applyPooling(
                int kernelSize, int stepSize, double[][] input, int outputRows, int outputCols, List<int[][]> lastPooledRows, List<int[][]> lastPooledCols) {
            return maxPooling(kernelSize, stepSize, input, outputRows, outputCols, lastPooledRows, lastPooledCols);
        }

    },

    AVERAGE {
        @Override
        public double[][] applyPooling(
                int kernelSize, int stepSize, double[][] input, int outputRows, int outputCols, List<int[][]> lastPooledRows, List<int[][]> lastPooledCols) {
            return averagePooling(kernelSize, stepSize, input, outputRows, outputCols, lastPooledRows, lastPooledCols);
        }
    };

    public abstract double[][] applyPooling(int kernelSize, int stepSize, double[][] input, int outputRows, int outputCols, List<int[][]> lastPooledRows, List<int[][]> lastPooledCols);

}
