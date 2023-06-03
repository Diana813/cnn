package dianaszczepankowska;

import dianaszczepankowska.activationfunction.ReLu;
import dianaszczepankowska.dataloader.Loader;
import dianaszczepankowska.dataloader.model.Image;
import dianaszczepankowska.layer.PoolingType;
import dianaszczepankowska.network.CnnModelBuilder;
import dianaszczepankowska.network.CnnNetwork;
import java.io.IOException;
import static java.util.Collections.shuffle;
import java.util.List;

public class Main {
    public static final String TRAIN_IMAGES_PATH = "src/main/java/dianaszczepankowska/dataloader/mnist_data/train-images-idx3-ubyte";
    public static final String TRAIN_LABELS_PATH = "src/main/java/dianaszczepankowska/dataloader/mnist_data/train-labels-idx1-ubyte";
    public static final String TEST_IMAGES_PATH = "src/main/java/dianaszczepankowska/dataloader/mnist_data/t10k-images-idx3-ubyte";
    public static final String TEST_LABELS_PATH = "src/main/java/dianaszczepankowska/dataloader/mnist_data/t10k-labels-idx1-ubyte";


    public static void main(String[] args) throws IOException {
        System.out.println("Pobieranie danych...");

        List<Image> imagesTrain = Loader.loadMnistImages(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);
        List<Image> imagesTest = Loader.loadMnistImages(TEST_IMAGES_PATH, TEST_LABELS_PATH);

        long SEED = 100;


        System.out.println("Liczba danych treningowych: " + imagesTrain.size());
        System.out.println("Liczba danych testowych: " + imagesTest.size());

        try {
            CnnModelBuilder builder = new CnnModelBuilder()
                    .addConvolutionLayer(30, 5, 1, 0.15, SEED)
                    .addPoolingLayer(3, 2, PoolingType.MAX)
                    .addDenseLayer(128, 0.15, new ReLu(), SEED)
                    .addDenseLayer(10, 0.15, new ReLu(), SEED);

            CnnNetwork cnn = builder.build();

            double rate = cnn.test(imagesTest);
            System.out.println("Współczynnik sukcesu przed treningiem: " + rate);

            int epochs = 30;

            for (int i = 0; i < epochs; i++) {
                shuffle(imagesTrain);
                cnn.train(imagesTrain);
                rate = cnn.test(imagesTest);
                System.out.println("Współczynnik sukcesu po rundzie " + i + ": " + rate);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
