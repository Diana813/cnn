package dianaszczepankowska.problem;

import dianaszczepankowska.problem.model.Image;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Loader {
    private static final int IMAGE_MAGIC_NUMBER = 2051;
    private static final int LABEL_MAGIC_NUMBER = 2049;
    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;


    public static List<Image> loadMnistImages(String imagesPath, String labelsPath) throws IOException {
        List<Image> images;

        try (DataInputStream imagesStream = readMnistDataFile(imagesPath, IMAGE_MAGIC_NUMBER)) {
            try (DataInputStream labelsStream = readMnistDataFile(labelsPath, LABEL_MAGIC_NUMBER)) {

                int numImages = imagesStream.readInt();
                int numRows = imagesStream.readInt();
                int numCols = imagesStream.readInt();

                if (numRows != IMAGE_HEIGHT || numCols != IMAGE_WIDTH) {
                    throw new RuntimeException("Invalid image size");
                }

                if (numImages != labelsStream.readInt()) {
                    throw new RuntimeException("Number of images and labels mismatch");
                }

                images = IntStream.range(0, numImages)
                        .mapToObj(i -> loadMnistImage(imagesStream, labelsStream))
                        .collect(Collectors.toList());
            }
        }

        return images;
    }

    private static DataInputStream readMnistDataFile(String path, int magicNumber) throws IOException {
        DataInputStream dataInputStream = new DataInputStream(new FileInputStream(path));
        int number = dataInputStream.readInt();
        if (number != magicNumber) {
            throw new RuntimeException("Invalid image file format");
        }
        return dataInputStream;
    }

    private static Image loadMnistImage(DataInputStream imagesStream, DataInputStream labelsStream) {
        try {
            byte[] data = new byte[IMAGE_HEIGHT * IMAGE_WIDTH];
            imagesStream.readFully(data);
            double[][] pixels = IntStream.range(0, IMAGE_HEIGHT)
                    .mapToObj(h -> IntStream.range(0, IMAGE_WIDTH)
                            .mapToDouble(w -> (double) (data[h * IMAGE_WIDTH + w] & 0xFF) / (256 * 1000))
                            .toArray())
                    .toArray(double[][]::new);
            int label = labelsStream.readUnsignedByte();
            return new Image(pixels, label);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

}
