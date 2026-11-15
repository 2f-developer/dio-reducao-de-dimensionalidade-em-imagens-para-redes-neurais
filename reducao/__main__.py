"""Uso: python -m reducao FOTO [--saida PASTA] [--limiar N]

Gera FOTO_cinza.png, FOTO_binaria.png e FOTO_comparacao.png (as três lado a lado).
"""

import argparse
from pathlib import Path

from PIL import Image

from .imagem import binarizar, carregar, limiar_otsu, painel, para_cinza


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reducao", description="Converte uma foto para cinza e para preto e branco."
    )
    parser.add_argument("foto", type=Path)
    parser.add_argument("--saida", type=Path, help="pasta de saída (padrão: a mesma da foto)")
    parser.add_argument("--limiar", type=int, help="0 a 255; sem ele, usa o limiar de Otsu")
    args = parser.parse_args(argv)

    if not args.foto.is_file():
        parser.error(f"arquivo não encontrado: {args.foto}")
    pasta = args.saida or args.foto.parent
    pasta.mkdir(parents=True, exist_ok=True)

    rgb = carregar(args.foto)
    cinza = para_cinza(rgb)
    limiar = args.limiar if args.limiar is not None else limiar_otsu(cinza)
    binaria = binarizar(cinza, limiar)

    base = pasta / args.foto.stem
    Image.fromarray(cinza).save(f"{base}_cinza.png")
    Image.fromarray(binaria).save(f"{base}_binaria.png")
    Image.fromarray(painel(rgb, cinza, binaria)).save(f"{base}_comparacao.png")
    print(f"{rgb.shape[1]}x{rgb.shape[0]}, limiar {limiar}: arquivos em {pasta}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
