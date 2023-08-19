package dianaszczepankowska.driver;

import static dianaszczepankowska.Main.SEED;
import dianaszczepankowska.method.activationfunction.ReLu;
import dianaszczepankowska.method.layer.PoolingType;
import dianaszczepankowska.method.network.CnnModelBuilder;
import dianaszczepankowska.problem.ImagePreprocessor;
import dianaszczepankowska.problem.model.Image;
import dianaszczepankowska.problem.model.ClothingCategory;
import dianaszczepankowska.method.network.CnnNetwork;
import java.io.File;
import java.util.List;
import java.util.Scanner;

public class InputHandler {

    private final Scanner scanner;
    private boolean isLastDenseLayer;


    public InputHandler(Scanner scanner) {
        this.scanner = scanner;
    }

    public CnnModelBuilder getCnnNetworkParams(CnnModelBuilder builder) {
        System.out.println("Ustaw parametry sieci");

        do {
            addConvolutionLayer(builder);
        } while (!scanner.nextLine().equals("q"));

        do {
            addDenseLayer(builder);
        } while (!isLastDenseLayer);

        return builder;

    }

    public void getUserInputImage(CnnNetwork cnn) {
        while (true) {
            System.out.println("Dodaj sciezke do zdjecia do klasyfikacji lub wpisz q aby zakonczyc:");

            String userInput = scanner.nextLine().replace("\"", "").toLowerCase();

            if (userInput.equals("q")) {
                break;
            }

            Image inputImage = ImagePreprocessor.processImagesFromFiles(List.of(new File(userInput))).get(0);

            double[] predictions = cnn.predictProbabilities(inputImage);
            System.out.println("Wynik klasyfikacji dla kazdej klasy:");
            for (ClothingCategory category : ClothingCategory.values()) {
                int classIndex = category.ordinal();
                System.out.println(category.name() + ": " + (predictions[classIndex] * 100) + "%");
            }

            System.out.println("Jesli chcesz zakonczyc wpisz: q");
            System.out.println("Jesli chcesz dodac kolejny obrazek wpisz dowolna litere");
        }

    }

    private void addConvolutionLayer(CnnModelBuilder builder) {
        int numFilters;
        int filterSize;
        int stepSize;
        double learningRate;
        boolean padding;


        try {
            System.out.println("DODAJ WARSTWE SPLOTOWA");

            System.out.println("Podaj liczbe filtrow: ");
            numFilters = scanner.nextInt();
            scanner.nextLine();

            System.out.println("Podaj rozmiar filtra: ");
            filterSize = scanner.nextInt();
            scanner.nextLine();

            System.out.println("Podaj wielkosc kroku: ");
            stepSize = scanner.nextInt();
            scanner.nextLine();

            System.out.println("Podaj learning rate: ");
            learningRate = scanner.nextDouble();
            scanner.nextLine();

            System.out.println("Dodac padding? (T/N) ");
            String paddingAnswer = scanner.nextLine();
            padding = paddingAnswer.equalsIgnoreCase("T");

            builder.addConvolutionLayer(numFilters, filterSize, stepSize, learningRate, padding, SEED);
            addPoolingLayer(builder);

            System.out.println("Jesli nie chcesz dodawac wiecej warstw splotowych wpisz q");
            System.out.println("Jesli chcesz dodac kolejna warstwe wpisz dowolna litere.");

        } catch (Exception e) {
            System.out.println("Parametry nieprawidlowe, nie mozna dodac warstwy splotowej. Sprobuj ponownie.");
            scanner.nextLine();
            addConvolutionLayer(builder);
        }

    }

    private void addPoolingLayer(CnnModelBuilder builder) {

        int kernelSize;
        int poolingStepSize;
        PoolingType type;

        try {
            System.out.println("DODAJ POOLING");
            System.out.println("Podaj rozmiar filtra: ");
            kernelSize = scanner.nextInt();
            scanner.nextLine();

            System.out.println("Podaj wielkosc kroku: ");
            poolingStepSize = scanner.nextInt();
            scanner.nextLine();

            System.out.println("Podaj typ poolingu: (max/average)");
            if (scanner.nextLine().equalsIgnoreCase("max")) {
                type = PoolingType.MAX;
            } else {
                type = PoolingType.AVERAGE;
            }

            builder.addPoolingLayer(kernelSize, poolingStepSize, type);
        } catch (Exception e) {
            System.out.println("Parametry nieprawidlowe, nie mozna dodac poolingu");
            scanner.nextLine();
            addPoolingLayer(builder);
        }
    }

    private void addDenseLayer(CnnModelBuilder builder) {

        int outLength;
        double denseLearningRate;
        System.out.println("DODAJ WARSTWE GESTA (w ostatniej warstwie gestej liczba neuronow musi wynosic 10)");

        try {
            System.out.println("Czy to ostatnia warstwa gesta? (T/N)");
            String isLastLayerAnswer = scanner.nextLine();
            isLastDenseLayer = isLastLayerAnswer.equalsIgnoreCase("T");

            System.out.println("Podaj liczbe neuronow: ");
            outLength = scanner.nextInt();

            if (isLastDenseLayer && outLength != 10) {
                System.out.println("W ostatniej warstwie gestej liczba neuronow musi wynosic 10");
                throw new IllegalArgumentException();
            }

            scanner.nextLine();
            System.out.println("Podaj learning rate: ");
            denseLearningRate = scanner.nextDouble();
            scanner.nextLine();

            builder.addDenseLayer(outLength, denseLearningRate, new ReLu(), SEED);

        } catch (Exception e) {
            System.out.println("Parametry nieprawidlowe, nie mozna dodac warstwy gestej");
            scanner.nextLine();
            addDenseLayer(builder);
        }

    }

}
