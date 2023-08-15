package dianaszczepankowska.problem;

import dianaszczepankowska.problem.model.Image;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.io.ByteArrayOutputStream;
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
                .map(ImagePreprocessor::processImageFromFile)
                .collect(Collectors.toList());
    }

    private static Image processImageFromFile(File file) {
        try {
            BufferedImage originalImage = ImageIO.read(file);
            BufferedImage grayScaleImage = convertToGrayScale(originalImage);
            BufferedImage scaledImage = scaleImage(grayScaleImage);
            byte[] ubyteData = convertImageToUbyte(scaledImage);
            double[][] pixelMatrix = convertToPixelMatrix(ubyteData);
            return ImageFactory.createFromPixelMatrix(pixelMatrix);
        } catch (IOException e) {
            throw new RuntimeException("Error processing image file: " + file.getName(), e);
        }
    }

    private static BufferedImage scaleImage(BufferedImage originalImage) {
        BufferedImage scaledImage = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = scaledImage.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(originalImage, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, null);
        g.dispose();
        return scaledImage;
    }

    public static double[][] convertToPixelMatrix(byte[] grayscaleData) {
        double[][] pixelMatrix = new double[IMAGE_HEIGHT][IMAGE_WIDTH];

        for (int y = 0; y < IMAGE_HEIGHT; y++) {
            for (int x = 0; x < IMAGE_WIDTH; x++) {
                int index = y * IMAGE_WIDTH + x;
                pixelMatrix[y][x] = (double) (grayscaleData[index] & 0xFF) / (256.0 * 1000.0);
            }
        }

        return pixelMatrix;
    }

    public static byte[] convertImageToUbyte(BufferedImage image) {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int rgb = image.getRGB(x, y);
                int red = (rgb >> 16) & 0xFF;
                int green = (rgb >> 8) & 0xFF;
                int blue = rgb & 0xFF;
                outputStream.write(red);
                outputStream.write(green);
                outputStream.write(blue);
            }
        }
        return outputStream.toByteArray();
    }

    private static BufferedImage convertToGrayScale(BufferedImage originalImage) {
        BufferedImage grayScaleImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        ColorConvertOp op = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null);
        op.filter(originalImage, grayScaleImage);
        return grayScaleImage;
    }

    public static class ImageFactory {
        public static Image createFromPixelMatrix(double[][] pixelMatrix) {
            return new Image(pixelMatrix, null);
        }
    }

}