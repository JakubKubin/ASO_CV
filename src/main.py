import argparse
import torch

from dataset_pipeline import get_dataloaders, DatasetConfig
from model_implementation import load_model
from model_train import train
from evaluation import evaluate_model, print_evaluation_results
from inference import process_image

def main():
    # Parsing argumentów wiersza poleceń
    parser = argparse.ArgumentParser(description='System detekcji produktów spożywczych')

    parser.add_argument('--mode', type=str, required=True, choices=['prepare', 'train', 'evaluate', 'infer'],
                      help='Tryb działania: prepare - przygotowanie danych, train - trening modelu, evaluate - ewaluacja, infer - detekcja na obrazie')

    parser.add_argument('--data_dir', type=str, default='data',
                      help='Katalog z danymi')
    parser.add_argument('--model_type', type=str, default='faster_rcnn', choices=['faster_rcnn', 'mask_rcnn'],
                      help='Typ modelu: faster_rcnn lub mask_rcnn')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Katalog wyjściowy na modele')
    parser.add_argument('--model_path', type=str,
                      help='Ścieżka do zapisanego modelu')
    parser.add_argument('--image_path', type=str,
                      help='Ścieżka do obrazu do analizy')
    parser.add_argument('--output_path', type=str,
                      help='Ścieżka do zapisania obrazu z detekcjami')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Liczba epok treningu')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Rozmiar batcha')
    parser.add_argument('--threshold', type=float, default=0.1,
                      help='Próg pewności dla detekcji')

    args = parser.parse_args()


    if args.mode == 'prepare':
        cfg = DatasetConfig(batch_size=args.batch_size)
        get_dataloaders(config=cfg)
        print("Dane zostały przygotowane")

    elif args.mode == 'train':
        print(f"Rozpoczynam trening modelu {args.model_type}...")
        train(
            model_type=args.model_type,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )

    elif args.mode == 'evaluate':
        # Ewaluacja modelu
        if not args.model_path:
            print("Błąd: Nie podano ścieżki do modelu")
            return

        # Inicjalizacja urządzenia
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Wczytanie modelu
        model = load_model(args.model_type, args.model_path)
        model.to(device)

        # Przygotowanie data loaderów
        _, val_loader = get_dataloaders(args.data_dir, args.batch_size)

        # Ewaluacja modelu
        results = evaluate_model(model, val_loader, device)

        # # Wyświetlenie wyników
        print_evaluation_results(results)

    elif args.mode == 'infer':
        # Inferecja na pojedynczym obrazie
        if not args.model_path:
            print("Błąd: Nie podano ścieżki do modelu")
            return

        if not args.image_path:
            print("Błąd: Nie podano ścieżki do obrazu")
            return

        model = load_model(args.model_type, args.model_path)

        process_image(
            model=model,
            image_path=args.image_path,
            output_path=args.output_path,
            threshold=args.threshold
        )

if __name__ == "__main__":
    main()