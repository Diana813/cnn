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
        } while (!scanner.nextLine().equals("q"));


        return builder;

    }

    public void getUserInputImage(CnnNetwork cnn) {
        do {
            System.out.println("Dodaj sciezkę do zdjęcia do klasyfikacji:");

            String imagePath = scanner.nextLine().replace("\"", "");
            Image inputImage = ImagePreprocessor.processImagesFromFiles(List.of(new File(imagePath))).get(0);

            double[] predictions = cnn.predictProbabilities(inputImage);
            System.out.println("Wynik klasyfikacji dla kazdej klasy:");
            for (ClothingCategory category : ClothingCategory.values()) {
                int classIndex = category.ordinal();
                System.out.println(category.name() + ": " + (predictions[classIndex] * 100) + "%");
            }

            System.out.println("Jesli chcesz zakonczyc wpisz: q");
            System.out.println("Jesli chcesz dodac kolejny obrazek wpisz dowolna litere");

        } while (!scanner.nextLine().equals("q"));
    }

    private void addConvolutionLayer(CnnModelBuilder builder) {
        int numFilters;
        int filterSize;
        int stepSize;
        double learningRate;
        boolean padding;

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

        System.out.println("Padding? (T/N) ");
        String paddingAnswer = scanner.nextLine();
        padding = paddingAnswer.equals("T");

        try {
            builder.addConvolutionLayer(numFilters, filterSize, stepSize, learningRate, padding, SEED);
            addPoolingLayer(builder);
            System.out.println("Jesli nie chcesz dodawac wiecej warstw splotowych wpisz q");
            System.out.println("Jesli chcesz dodac kolejna warstwe wpisz dowolna litere.");
        } catch (Exception e) {
            System.out.println("Parametry nieprawidlowe, nie mozna dodac warstwy splotowej");
            addConvolutionLayer(builder);
        }

    }

    private void addPoolingLayer(CnnModelBuilder builder) {

        int kernelSize;
        int poolingStepSize;
        PoolingType type;

        System.out.println("DODAJ POOLING(jesli nie chcesz dodawac poolingu wpisz q)");
        System.out.println("Podaj rozmiar filtra: ");
        kernelSize = scanner.nextInt();
        scanner.nextLine();

        System.out.println("Podaj wielkosc kroku: ");
        poolingStepSize = scanner.nextInt();
        scanner.nextLine();

        System.out.println("Podaj typ poolingu: (max/average)");
        if (scanner.nextLine().equals("max")) {
            type = PoolingType.MAX;
        } else {
            type = PoolingType.AVERAGE;
        }

        try {
            builder.addPoolingLayer(kernelSize, poolingStepSize, type);
        } catch (Exception e) {
            System.out.println("Parametry nieprawidlowe, nie mozna dodac poolingu");
            addPoolingLayer(builder);
        }
    }

    private void addDenseLayer(CnnModelBuilder builder) {

        int outLength;
        double denseLearningRate;
        System.out.println("DODAJ WARSTWE GESTA (jesli to ostatnia z warstw gestych liczba neuronow musi wynosic 10)");

        System.out.println("Podaj liczbe neuronow: ");
        outLength = scanner.nextInt();
        scanner.nextLine();
        System.out.println("Podaj learning rate: ");
        denseLearningRate = scanner.nextDouble();
        scanner.nextLine();

        try {

            builder.addDenseLayer(outLength, denseLearningRate, new ReLu(), SEED);
            System.out.println("Jesli nie chcesz dodawas wiecej warstw gestych wpisz q");
            System.out.println("Jesli chcesz dodas kolejna warstwe wpisz dowolna litere.");
        } catch (Exception e) {
            System.out.println("Parametry nieprawidlowe, nie mozna dodac warstwy gestej");
            addDenseLayer(builder);
        }

    }

}
