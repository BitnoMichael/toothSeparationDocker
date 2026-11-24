"""
utils/Flops_and_Params.py

Замер FLOPs и параметров для модели MeshSegNet.

ВНИМАНИЕ: этот скрипт не является частью обучающего/тестового пайплайна и
вызывается вручную. Для его работы требуются пакеты `torchsummary`, `thop` и
`torchstat`, которых НЕТ в requirements.txt (чтобы не раздувать образ).
Перед запуском установите их вручную:

    pip install torchsummary thop torchstat

Если CUDA недоступна, скрипт использует CPU автоматически.
"""
import torch

# Импорты тяжёлых метрических библиотек вынесены в guarded-блок, чтобы
# скрипт не падал с ImportError при простом `python utils/Flops_and_Params.py`
# в образе, где эти библиотеки не установлены.
try:
    from torchsummary import summary
    from thop import profile
    _METRIC_LIBS_AVAILABLE = True
except ImportError as _e:
    _METRIC_LIBS_AVAILABLE = False
    _MISSING_IMPORT = str(_e)

from networks.MeshSegNet import MeshSegNet


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MeshSegNet().to(device)
    model.eval()

    # MeshSegNet ожидает на вход форму, использовавшуюся в экспериментах: (2, 24, 300).
    data_shape = (2, 24, 300)

    if not _METRIC_LIBS_AVAILABLE:
        print(
            f"\n[SKIP] Метрические библиотеки недоступны ({_MISSING_IMPORT}).\n"
            "Установите их вручную для замера FLOPs/Params:\n"
            "    pip install torchsummary thop torchstat\n"
            "Простое число параметров модели приведено ниже."
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {n_params}")
        return

    # torchsummary требует input_shape без batch-размерности.
    summary(model, data_shape, device=str(device))

    sample = torch.rand(1, *data_shape, device=device)
    flops, params = profile(model, inputs=(sample,), verbose=False)
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")


if __name__ == "__main__":
    main()
