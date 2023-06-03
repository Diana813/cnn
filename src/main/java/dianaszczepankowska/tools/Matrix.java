package dianaszczepankowska.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class Matrix {

    public static double[][] add(double[][] a, double[][] b) {
        return IntStream.range(0, a.length)
                .mapToObj(i -> IntStream.range(0, a[0].length)
                        .mapToDouble(j -> a[i][j] + b[i][j])
                        .toArray())
                .toArray(double[][]::new);

    }

    public static double[] add(double[] a, double[] b) {
        return IntStream.range(0, a.length)
                .mapToDouble(i -> a[i] + b[i])
                .toArray();
    }

    public static double[][] multiply(double[][] a, double scalar) {
        return Arrays.stream(a)
                .map(row -> Arrays.stream(row)
                        .map(element -> element * scalar)
                        .toArray())
                .toArray(double[][]::new);
    }

    public static double[] multiply(double[] a, double scalar) {
        return Arrays.stream(a)
                .map(element -> element * scalar)
                .toArray();
    }

    public static double[][] convolve(double[][] input, double[][] filter, int stepSize) {
        int outRows = (input.length - filter.length) / stepSize + 1;
        int outCols = (input[0].length - filter[0].length) / stepSize + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for (int i = 0; i <= inRows - fRows; i += stepSize) {

            outCol = 0;

            for (int j = 0; j <= inCols - fCols; j += stepSize) {

                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        double value = filter[x][y] * input[i + x][j + y];
                        sum += value;
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;

        }

        return output;
    }

    public static double[][] fullConvolve(double[][] input, double[][] filter) {

        int outRows = (input.length + filter.length) + 1;
        int outCols = (input[0].length + filter[0].length) + 1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow = 0;
        int outCol;

        for (int i = -fRows + 1; i < inRows; i++) {

            outCol = 0;

            for (int j = -fCols + 1; j < inCols; j++) {

                double sum = 0.0;

                for (int x = 0; x < fRows; x++) {
                    for (int y = 0; y < fCols; y++) {
                        int inputRowIndex = i + x;
                        int inputColIndex = j + y;

                        if (inputRowIndex >= 0 && inputColIndex >= 0 && inputRowIndex < inRows && inputColIndex < inCols) {
                            double value = filter[x][y] * input[inputRowIndex][inputColIndex];
                            sum += value;
                        }
                    }
                }

                output[outRow][outCol] = sum;
                outCol++;
            }

            outRow++;

        }

        return output;

    }

    public static double[][] expandArray(double[][] input, int stepSize) {
        if (stepSize == 1) {
            return input;
        }

        int outRows = (input.length - 1) * stepSize + 1;
        int outCols = (input[0].length - 1) * stepSize + 1;

        double[][] output = new double[outRows][outCols];

        IntStream.range(0, input.length)
                .forEach(i -> IntStream.range(0, input[0].length)
                        .forEach(j -> output[i * stepSize][j * stepSize] = input[i][j]));

        return output;
    }

    public static double[][] flipHorizontal(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(array[i], 0, output[rows - i - 1], 0, cols);
        }
        return output;
    }

    public static double[][] flipVertical(double[][] array) {
        int rows = array.length;
        int cols = array[0].length;

        double[][] output = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[i][cols - j - 1] = array[i][j];
            }
        }
        return output;
    }

    public static double[] matrixToVector(List<double[][]> input) {
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length * rows * cols];

        int i = 0;

        for (double[][] doubles : input) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    vector[i] = doubles[r][c];
                    i++;
                }
            }
        }

        return vector;
    }

    public static List<double[][]> vectorToMatrix(double[] input, int length, int rows, int cols) {

        List<double[][]> out = new ArrayList<>();

        int i = 0;

        for (int l = 0; l < length; l++) {
            double[][] matrix = new double[rows][cols];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    matrix[r][c] = input[i];
                    i++;
                }
            }
            out.add(matrix);
        }
        return out;
    }

    public static double[][] maxPooling(int kernelSize, int stepSize, double[][] input, int outputRows, int outputCols, List<int[][]> lastPooledRows, List<int[][]> lastPooledCols) {
        double[][] output = new double[outputRows][outputCols];

        int[][] maxRows = new int[outputRows][outputCols];
        int[][] maxCols = new int[outputRows][outputCols];

        for (int r = 0; r < outputRows; r += stepSize) {
            for (int c = 0; c < outputCols; c += stepSize) {
                double max = 0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                for (int k = 0; k < kernelSize; k++) {
                    for (int i = 0; i < kernelSize; i++) {
                        if (max < input[r + k][c + i]) {
                            max = input[r + k][c + i];

                            maxRows[r][c] = r + k;
                            maxCols[r][c] = c + i;
                        }
                    }
                }
                output[r][c] = max;
            }
        }

        lastPooledRows.add(maxRows);
        lastPooledCols.add(maxCols);
        return output;
    }

    public static double[][] averagePooling(int kernelSize, int stepSize, double[][] input, int outputRows, int outputCols, List<int[][]> lastPooledRows, List<int[][]> lastPooledCols) {
        double[][] output = new double[outputRows][outputCols];

        int[][] pooledRows = new int[outputRows][outputCols];
        int[][] pooledCols = new int[outputRows][outputCols];

        for (int r = 0; r < outputRows; r += stepSize) {
            for (int c = 0; c < outputCols; c += stepSize) {
                double sum = 0;
                int count = 0;

                for (int k = 0; k < kernelSize; k++) {
                    for (int i = 0; i < kernelSize; i++) {
                        int rowIndex = r + k;
                        int colIndex = c + i;

                        if (rowIndex < input.length && colIndex < input[0].length) {
                            sum += input[rowIndex][colIndex];
                            count++;
                        }
                    }
                }

                double average = sum / count;
                output[r][c] = average;

                pooledRows[r][c] = r;
                pooledCols[r][c] = c;
            }
        }

        lastPooledRows.add(pooledRows);
        lastPooledCols.add(pooledCols);
        return output;
    }

}
