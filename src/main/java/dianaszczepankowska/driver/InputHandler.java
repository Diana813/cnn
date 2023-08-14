package dianaszczepankowska.driver;

import static dianaszczepankowska.dataloader.ImagePreprocessor.processImagesFromFiles;
import dianaszczepankowska.dataloader.model.ClothingCategory;
import dianaszczepankowska.dataloader.model.Image;
import dianaszczepankowska.network.CnnNetwork;
import java.io.File;
import java.util.List;
import java.util.Scanner;

public class InputHandler {

    public void getUserInputImage(CnnNetwork cnn){
        Scanner scanner = new Scanner(System.in);

        while (!scanner.nextLine().equals("q")){
            System.out.println("Dodaj ścieżkę do zdjęcia do klasyfikacji:");

            String imagePath = scanner.nextLine();
            Image inputImage = processImagesFromFiles(List.of(new File(imagePath))).get(0);

            double[] predictions = cnn.predictProbabilities(inputImage);
            System.out.println("Wynik klasyfikacji dla każdej klasy:");
            for (ClothingCategory category : ClothingCategory.values()) {
                int classIndex = category.ordinal();
                System.out.println(category.name() + ": " + (predictions[classIndex] * 100) + "%");
            }

            System.out.println("Jeśli chcesz zakończyć wpisz: q");
        }
    }
}
