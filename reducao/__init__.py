"""Conversão de imagens coloridas para níveis de cinza e para preto e branco."""

from .imagem import binarizar, limiar_otsu, para_cinza

__all__ = ["binarizar", "limiar_otsu", "para_cinza"]
