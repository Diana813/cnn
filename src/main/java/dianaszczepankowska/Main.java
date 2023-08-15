package dianaszczepankowska;

import dianaszczepankowska.problem.model.Image;
import dianaszczepankowska.method.activationfunction.ReLu;
import dianaszczepankowska.problem.Loader;
import dianaszczepankowska.driver.InputHandler;
import dianaszczepankowska.method.layer.PoolingType;
import dianaszczepankowska.method.network.CnnModelBuilder;
import dianaszczepankowska.method.network.CnnNetwork;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import static java.util.Collections.shuffle;
import java.util.List;
import java.util.Scanner;


public class Main {
    public static final String TRAIN_IMAGES_PATH = "src/main/java/dianaszczepankowska/problem/mnistdata/train-images-idx3-ubyte";
    public static final String TRAIN_LABELS_PATH = "src/main/java/dianaszczepankowska/problem/mnistdata/train-labels-idx1-ubyte";
    public static final String TEST_IMAGES_PATH = "src/main/java/dianaszczepankowska/problem/mnistdata/t10k-images-idx3-ubyte";
    public static final String TEST_LABELS_PATH = "src/main/java/dianaszczepankowska/problem/mnistdata/t10k-labels-idx1-ubyte";
    public static long SEED = 100;


    public static void main(String[] args) throws IOException {

        PrintWriter fileOut = new PrintWriter(new BufferedWriter(new FileWriter("results.txt", true)));
        InputHandler inputHandler = new InputHandler(new Scanner(System.in));


        System.out.println("Pobieranie danych...");

        List<Image> imagesTrain = Loader.loadMnistImages(TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH);
        List<Image> imagesTest = Loader.loadMnistImages(TEST_IMAGES_PATH, TEST_LABELS_PATH);


        System.out.println("Liczba danych treningowych: " + imagesTrain.size());
        System.out.println("Liczba danych testowych: " + imagesTest.size());


        try {


            CnnModelBuilder builder = new CnnModelBuilder()
                    .addConvolutionLayer(20, 7, 1, 0.25, true, SEED)
                    .addPoolingLayer(3, 1, PoolingType.MAX)
                    .addConvolutionLayer(10, 5, 1, 0.25, true, SEED)
                    .addPoolingLayer(3, 1, PoolingType.MAX)
                    .addDenseLayer(150, 0.25, new ReLu(), SEED)
                    .addDenseLayer(10, 0.25, new ReLu(), SEED)
                    .addSoftMaxLayer(10);

            //builder = inputHandler.getCnnNetworkParams(builder);
           /// builder.addSoftMaxLayer(10);


            CnnNetwork cnn = builder.build();

            double initialRate = cnn.test(imagesTest);
            fileOut.println("WSPOLCZYNNIK SUKCESU PRZED TRENINGIEM: " + initialRate);
            System.out.println("Współczynnik sukcesu przed treningiem: " + initialRate);

            String networkSettings = builder.toString();
            fileOut.println("USTAWIENIA SIECI: " + networkSettings);

            int epochs = 5;

            for (int i = 0; i < epochs; i++) {
                shuffle(imagesTrain);
                cnn.train(imagesTrain);
                double rate = cnn.test(imagesTest);
                System.out.println("Współczynnik sukcesu po rundzie " + i + ": " + rate);

                fileOut.close();
                fileOut = new PrintWriter(new BufferedWriter(new FileWriter("results.txt", true)));
                fileOut.println("WSPOLCZYNNIK SUKCESU PO RUNDZIE " + i + ": " + rate);
                fileOut.flush();
            }
            fileOut.println();

            inputHandler.getUserInputImage(cnn);

        } catch (Exception e) {
            e.printStackTrace();
        }

        fileOut.close();

    }


}
