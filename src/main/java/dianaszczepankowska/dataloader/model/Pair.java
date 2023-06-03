package dianaszczepankowska.dataloader.model;

public record Pair<T, U>(T first, U second) {
    public static <F, S> Pair<F, S> of(F first, S second) {
        return new Pair<>(first, second);
    }
}