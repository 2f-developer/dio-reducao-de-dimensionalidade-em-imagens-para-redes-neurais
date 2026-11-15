"""Redução de dimensionalidade: RGB (3 canais) → cinza (1 canal, 0 a 255) → binária (0 ou 255).

A conversão é feita à mão com NumPy. O Pillow só entra para ler e gravar arquivos.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Pesos da luminância (ITU-R BT.601): o olho é mais sensível ao verde e menos ao azul.
PESOS_RGB = np.array([0.299, 0.587, 0.114])


def para_cinza(rgb: np.ndarray) -> np.ndarray:
    """Converte uma imagem RGB (altura, largura, 3) em cinza (altura, largura), 0 a 255."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Esperava uma imagem RGB (altura, largura, 3), recebi forma {rgb.shape}.")
    cinza = rgb.astype(np.float64) @ PESOS_RGB
    return np.clip(np.rint(cinza), 0, 255).astype(np.uint8)


def limiar_otsu(cinza: np.ndarray) -> int:
    """Escolhe o limiar que melhor separa a imagem em dois grupos (método de Otsu).

    Testa os 256 cortes possíveis e fica com o que maximiza a variância entre as
    classes "escura" e "clara". Funciona bem quando o histograma tem dois picos.
    """
    histograma = np.bincount(cinza.ravel(), minlength=256).astype(np.float64)
    total = histograma.sum()
    if total == 0:
        raise ValueError("Imagem vazia.")
    niveis = np.arange(256)
    peso_escuro = np.cumsum(histograma)
    soma_escuro = np.cumsum(histograma * niveis)
    peso_claro = total - peso_escuro
    with np.errstate(divide="ignore", invalid="ignore"):
        media_escuro = soma_escuro / peso_escuro
        media_claro = (soma_escuro[-1] - soma_escuro) / peso_claro
        variancia_entre = peso_escuro * peso_claro * (media_escuro - media_claro) ** 2
    if np.all(np.isnan(variancia_entre)):
        return int(cinza.flat[0])  # imagem de uma cor só: não há o que separar
    # Se houver um intervalo de tons sem nenhum pixel entre os dois grupos, todos os
    # cortes dentro dele empatam; o meio do empate fica equidistante dos dois grupos.
    maximo = np.nanmax(variancia_entre)
    empatados = np.flatnonzero(variancia_entre >= maximo * (1 - 1e-9))
    return int(round(empatados.mean()))


def binarizar(cinza: np.ndarray, limiar: int | None = None) -> np.ndarray:
    """Pixels acima do limiar viram 255 (branco); os demais, 0 (preto).

    Sem limiar informado, usa o de Otsu, calculado para a própria imagem.
    """
    if cinza.ndim != 2:
        raise ValueError("Binarize a imagem já convertida para cinza.")
    if limiar is None:
        limiar = limiar_otsu(cinza)
    if not 0 <= limiar <= 255:
        raise ValueError(f"O limiar precisa estar entre 0 e 255, recebi {limiar}.")
    return np.where(cinza > limiar, 255, 0).astype(np.uint8)


def carregar(caminho: str | Path) -> np.ndarray:
    with Image.open(caminho) as imagem:
        return np.asarray(imagem.convert("RGB"))


def painel(*imagens: np.ndarray, margem: int = 8) -> np.ndarray:
    """Coloca as imagens lado a lado (as de 1 canal viram RGB) para comparar."""
    rgb = [np.repeat(im[..., None], 3, axis=2) if im.ndim == 2 else im for im in imagens]
    altura = max(im.shape[0] for im in rgb)
    largura = sum(im.shape[1] for im in rgb) + margem * (len(rgb) - 1)
    saida = np.full((altura, largura, 3), 255, dtype=np.uint8)
    x = 0
    for im in rgb:
        saida[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1] + margem
    return saida
