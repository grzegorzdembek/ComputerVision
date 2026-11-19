# lab5: trening sieci YOLO (fine-tuning) oraz ewaluacja na zbiorze testowym
from pathlib import Path

from ultralytics import YOLO


def trainYoloModel(dataset_yaml_path: Path, project_directory: Path) -> Path:
    """
    Laduje lekki model bazowy yolo11n.pt i dotrenowuje go na naszym zbiorze czerwonych kulek.
    Zwraca sciezke do pliku z najlepszymi wagami (best.pt).
    """
    # pobieramy bazowy model nano (najlzejsza architektura, idealna pod wysoki FPS na kamerze)
    base_model = YOLO("yolo11n.pt")

    base_model.train(
        data=str(dataset_yaml_path),
        epochs=30,                       # liczba pelnych przejsc przez zbior treningowy
        imgsz=416,                       # rozmiar wejsciowy sieci (416x416 - dobry kompromis jakosc/FPS)
        batch=16,                        # ile obrazkow karta GPU przetwarza naraz w jednym kroku
        device=0,                        # 0 oznacza pierwsza karte NVIDIA (GTX 1650 Ti)
        amp=False,                       # wylaczamy FP16 (zapobiega bledom NaN na kartach GTX 1650/1660)
        workers=0,                       # bezpieczne wczytywanie danych w jednym procesie na Windowsie
        project=str(project_directory),  # folder glowny na wyniki treningu
        name="red_ball_model",           # podfolder konkretnego eksperymentu
        exist_ok=True,                   # nadpisuje folder przy ponownym uruchomieniu zamiast tworzyc train2, train3...
    )

    best_weights_path = project_directory / "red_ball_model" / "weights" / "best.pt"
    return best_weights_path


def evaluateOnTestSet(weights_path: Path, test_images_directory: Path, project_directory: Path) -> None:
    """
    Uruchamia wytrenowany model na nieznanych wczesniej obrazach testowych
    i zapisuje obrazy z naniesionymi predykcjami na dysk.
    """
    trained_model = YOLO(str(weights_path))

    trained_model.predict(
        source=str(test_images_directory),
        conf=0.50,                       # prog pewnosci: pokazujemy wykrycia powyzej 50%
        imgsz=416,
        device=0,
        save=True,                       # zapisuje obrazy z narysowanymi ramkami
        project=str(project_directory),
        name="test_predictions",
        exist_ok=True,
    )


def main() -> None:
    lab_directory = Path(__file__).resolve().parent
    dataset_yaml_path = lab_directory / "dataset.yaml"
    runs_directory = lab_directory / "runs"
    test_images_directory = lab_directory / "dataset" / "images" / "test"

    if not dataset_yaml_path.exists():
        raise FileNotFoundError("Brak pliku dataset.yaml. Uruchom najpierw lab05/build_dataset.py!")

    print("Rozpoczynam trening modelu YOLO na GPU...")
    best_weights_path = trainYoloModel(dataset_yaml_path, runs_directory)

    print(f"Trening zakonczony! Najlepsze wagi zapisano w: {best_weights_path}")
    print("Uruchamiam test modelu na zbiorze testowym...")
    evaluateOnTestSet(best_weights_path, test_images_directory, runs_directory)

    print(f"Gotowe! Sprawdz wyniki w folderze: {runs_directory / 'test_predictions'}")


if __name__ == "__main__":
    main()