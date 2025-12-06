import torch
import torch.nn as nn

# Import the model architectures
from alg.architectures.cnn import BaseCnnActorCritic
from alg.architectures.resnet import BaseResNetActorCritic
from alg.architectures.transformer import BaseTransformerActorCritic
from alg.architectures.configs import (
    CnnSActorCritic, CnnLActorCritic,
    ResNetSActorCritic, ResNetLActorCritic,
    TransformerSActorCritic, TransformerLActorCritic
)


def count_parameters(model, trainable_only=True, verbose=True):
    """
    Oblicza liczbę parametrów w modelu PyTorch.
    
    Args:
        model: model PyTorch (nn.Module)
        trainable_only: jeśli True, liczy tylko parametry z requires_grad=True
        verbose: jeśli True, wypisuje szczegółowe informacje
    
    Returns:
        total_params: całkowita liczba parametrów
        trainable_params: liczba uczących się parametrów
    """
    if trainable_only:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
        non_trainable_params = 0
    
    # Liczba parametrów w przystępnym formacie
    def format_num(num):
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        else:
            return str(num)
    
    if verbose:
        print("=" * 50)
        print(f"Model: {model.__class__.__name__}")
        print(f"Liczba parametrów całkowita: {format_num(total_params)} ({total_params:,})")
        
        if non_trainable_params > 0:
            print(f"Parametry uczące się: {format_num(total_params)}")
            print(f"Parametry nieuczące się: {format_num(non_trainable_params)}")
        
        # Szczegółowa analiza po warstwach
        print("-" * 50)
        print("Szczegółowo po warstwach:")
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                trainable_module = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"  {name}: {format_num(module_params)} ({module_params:,})")
        
        print("=" * 50)
    
    return total_params, total_params - non_trainable_params


# Wersja bardziej szczegółowa - z podziałem na typy warstw
def count_parameters_detailed(model, print_table=True):
    """
    Szczegółowa analiza parametrów modelu z podziałem na typy warstw.
    
    Args:
        model: model PyTorch
        print_table: jeśli True, wypisuje tabelę z podsumowaniem
    
    Returns:
        dict: słownik ze statystykami parametrów
    """
    total_params = 0
    trainable_params = 0
    layer_stats = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        
        # Grupowanie po typie warstwy
        layer_type = name.split('.')[0] if '.' in name else 'other'
        if 'weight' in name or 'bias' in name:
            # Określ typ warstwy na podstawie nazwy
            if 'conv' in name.lower():
                layer_type = 'Conv'
            elif 'linear' in name.lower() or 'fc' in name.lower():
                layer_type = 'Linear'
            elif 'norm' in name.lower() or 'bn' in name.lower():
                layer_type = 'Normalization'
            elif 'embed' in name.lower():
                layer_type = 'Embedding'
            elif 'transformer' in name.lower():
                layer_type = 'Transformer'
            elif 'pos_embed' in name:
                layer_type = 'PositionalEmbedding'
        
        if layer_type not in layer_stats:
            layer_stats[layer_type] = {'total': 0, 'trainable': 0}
        
        layer_stats[layer_type]['total'] += param_count
        if param.requires_grad:
            layer_stats[layer_type]['trainable'] += param_count
    
    if print_table:
        print("\n" + "=" * 70)
        print(f"{'ANALIZA PARAMETRÓW MODELU':^70}")
        print("=" * 70)
        print(f"Model: {model.__class__.__name__}")
        print("-" * 70)
        print(f"{('Typ warstwy'):<20} {('Wszystkie'):>12} {('Uczące się'):>12} {'%':>8}")
        print("-" * 70)
        
        for layer_type, stats in sorted(layer_stats.items()):
            all_params = stats['total']
            trainable = stats['trainable']
            percentage = (trainable / total_params * 100) if total_params > 0 else 0
            print(f"{layer_type:<20} {all_params:>12,} {trainable:>12,} {percentage:>7.1f}%")
        
        print("-" * 70)
        print(f"{('RAZEM'):<20} {total_params:>12,} {trainable_params:>12,} {100:>7.1f}%")
        print("=" * 70)
        
        # Podsumowanie w przystępnym formacie
        print(f"\nPodsumowanie:")
        print(f"  Całkowita liczba parametrów: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Parametry uczące się: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  Stosunek uczących się: {trainable_params/total_params*100:.1f}%")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_stats': layer_stats
    }


# Przykład użycia z Twoimi modelami
def analyze_model_for_board_size(board_size=9, channels=2):
    """
    Analizuje wszystkie modele dla danej wielkości planszy.
    
    Args:
        board_size: rozmiar planszy (9 lub 15)
        channels: liczba kanałów wejściowych
    """
    action_dim = board_size * board_size
    obs_shape = (channels, board_size, board_size)
    
    print(f"\n{'='*80}")
    print(f"ANALIZA DLA PLANSZY {board_size}x{board_size} ({channels} kanały wejściowe)")
    print(f"{'='*80}")
    
    # Tworzymy wszystkie modele
    models = {
        'CNN S': CnnSActorCritic(obs_shape, action_dim),
        'CNN L': CnnLActorCritic(obs_shape, action_dim),
        'ResNet S': ResNetSActorCritic(obs_shape, action_dim),
        'ResNet L': ResNetLActorCritic(obs_shape, action_dim),
        'Transformer S': TransformerSActorCritic(obs_shape, action_dim),
        'Transformer L': TransformerLActorCritic(obs_shape, action_dim),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{'#'*60}")
        print(f"Analizuję: {name}")
        print(f"{'#'*60}")
        stats = count_parameters_detailed(model, print_table=True)
        results[name] = stats
    
    # Porównanie wszystkich modeli
    print(f"\n{'='*80}")
    print(f"{('PORÓWNANIE WSZYSTKICH MODELI'):^80}")
    print(f"{'='*80}")
    print(f"{('Model'):<20} {'Parametry':>12} {'Uczące się':>12} {'Rozmiar':>10}")
    print(f"{(''):<20} {'(wszystkie)':>12} {'':>12} {'(MB)':>10}")
    print(f"{'-'*80}")
    
    for name in models.keys():
        stats = results[name]
        total_mb = stats['total_params'] * 4 / (1024**2)  # 4 bajty na float32
        print(f"{name:<20} {stats['total_params']:>12,} {stats['trainable_params']:>12,} {total_mb:>9.2f}MB")
    
    print(f"{'='*80}")
    
    return results


# Szybka funkcja do obliczania tylko całkowitej liczby parametrów
def get_total_params(model):
    """Zwraca całkowitą liczbę parametrów w modelu."""
    return sum(p.numel() for p in model.parameters())

def get_trainable_params(model):
    """Zwraca liczbę uczących się parametrów w modelu."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Przykład użycia
if __name__ == "__main__":
    # Analiza dla planszy 9x9
    results_9x9 = analyze_model_for_board_size(board_size=9, channels=2)
    
    # Analiza dla planszy 15x15
    results_15x15 = analyze_model_for_board_size(board_size=15, channels=2)
    
    # Proste użycie dla pojedynczego modelu
    print("\nPrzykład prostej analizy pojedynczego modelu:")
    # This example was given by the user, but since the model classes are imported from configs.py
    # and not directly available here without an explicit import of the specific class, 
    # it's better to use one of the imported models for consistency.
    model = CnnSActorCritic((2, 9, 9), 81)
    total, trainable = count_parameters(model, verbose=True)
    print(f"Całkowita liczba parametrów: {total:,}")
    print(f"Uczące się parametry: {trainable:,}")
