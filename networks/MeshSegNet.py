"""
networks/MeshSegNet.py

Заглушка модели MeshSegNet, чтобы `utils/Flops_and_Params.py` мог импортировать
`from networks.MeshSegNet import MeshSegNet` без ошибки ModuleNotFoundError.

Реальная архитектура MeshSegNet в этом репозитории отсутствует. Эта заглушка
реализует минимальную совместимую сигнатуру: принимает тензор формы
[B, 2, 24, 300] (исходная форма входа из экспериментов) и возвращает
[B, num_classes, ...]. Используется ТОЛЬКО для замера FLOPs/Params; обучение
или инференс на реальных данных с этой заглушкой не имеют смысла.
"""
import torch
import torch.nn as nn


class MeshSegNet(nn.Module):
    """Минимальная заглушка MeshSegNet для запуска utils/Flops_and_Params.py."""

    def __init__(self, num_classes=17, num_channels=24):
        super().__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels

        # Лёгкая свёрточная «заглушка», сохраняющая входную форму каналов,
        # чтобы `torchsummary`/`thop` могли пройти по графу и посчитать
        # FLOPs/Params. Не является реальной моделью MeshSegNet.
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Ожидается вход [B, 2, num_channels=24, T=300] (как в экспериментах).
        # Сводим две «координатные» оси в одну канальную: [B, 2*24, 300].
        if x.dim() == 4:
            b, c2, ch, t = x.shape
            x = x.reshape(b, c2 * ch, t)
        return self.conv(x)
