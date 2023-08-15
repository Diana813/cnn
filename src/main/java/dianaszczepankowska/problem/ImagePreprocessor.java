package dianaszczepankowska.problem;

import dianaszczepankowska.problem.model.Image;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

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
            BufferedImage scaledImage = scaleImage(originalImage);
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

    public static class ImageFactory {
        public static Image createFromPixelMatrix(double[][] pixelMatrix) {
            return new Image(pixelMatrix, null);
        }
    }

    public static void displayProcessedImages(List<Image> processedImages) {
        JFrame frame = new JFrame();
        frame.setLayout(new GridLayout(1, processedImages.size()));

        for (Image image : processedImages) {
            BufferedImage bufferedImage = convertToBufferedImage(image.pixels());
            ImageIcon imageIcon = new ImageIcon(bufferedImage);
            JLabel label = new JLabel(imageIcon);
            frame.add(label);
        }

        frame.pack();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    private static BufferedImage convertToBufferedImage(double[][] pixelMatrix) {
        int height = pixelMatrix.length;
        int width = pixelMatrix[0].length;
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int grayscaleValue = (int) (pixelMatrix[y][x] * 255);
                int rgb = new Color(grayscaleValue, grayscaleValue, grayscaleValue).getRGB();
                image.setRGB(x, y, rgb);
            }
        }

        return image;
    }
}