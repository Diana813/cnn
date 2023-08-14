package dianaszczepankowska.problem;

import dianaszczepankowska.problem.model.Image;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;

public class ImagePreprocessor {

    private static final int IMAGE_WIDTH = 28;
    private static final int IMAGE_HEIGHT = 28;

    public static List<Image> processImagesFromFiles(List<File> imageFiles) {
        return imageFiles.stream()
                .map(file -> {
                    try {
                        BufferedImage originalImage = ImageIO.read(file);
                        return new Image(convertToGrayscaleMatrix(scaleImage(originalImage)), null);
                    } catch (IOException e) {
                        throw new RuntimeException("Error processing image file: " + file.getName(), e);
                    }
                })
                .collect(Collectors.toList());
    }

    private static BufferedImage scaleImage(BufferedImage originalImage) {
        BufferedImage scaledImage = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = scaledImage.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(originalImage, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, null);
        g.dispose();
        return scaledImage;
    }

    private static double[][] convertToGrayscaleMatrix(BufferedImage scaledImage) {
        int IMAGE_WIDTH = scaledImage.getWidth();
        int IMAGE_HEIGHT = scaledImage.getHeight();

        double[][] imageData = new double[IMAGE_HEIGHT][IMAGE_WIDTH];
        int[] pixelData = scaledImage.getRGB(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, null, 0, IMAGE_WIDTH);

        for (int h = 0; h < IMAGE_HEIGHT; h++) {
            for (int w = 0; w < IMAGE_WIDTH; w++) {
                int pixelValue = pixelData[h * IMAGE_WIDTH + w] & 0xFF;
                imageData[h][w] = (double) pixelValue / 255.0;
            }
        }

        return imageData;
    }
}