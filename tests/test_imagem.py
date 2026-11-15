import numpy as np
import pytest

from reducao.__main__ import main
from reducao.imagem import binarizar, carregar, limiar_otsu, painel, para_cinza


def test_cinza_usa_os_pesos_de_luminancia():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    assert para_cinza(rgb).tolist() == [[76, 150, 29, 255, 0]]


def test_cinza_reduz_tres_canais_para_um():
    rgb = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    cinza = para_cinza(rgb)
    assert cinza.shape == (20, 30)
    assert cinza.dtype == np.uint8


def test_cinza_recusa_imagem_que_nao_e_rgb():
    with pytest.raises(ValueError):
        para_cinza(np.zeros((4, 4), dtype=np.uint8))


def test_binaria_so_tem_preto_e_branco_e_respeita_o_limiar():
    cinza = np.array([[0, 100, 128, 129, 255]], dtype=np.uint8)
    assert binarizar(cinza, 128).tolist() == [[0, 0, 0, 255, 255]]
    assert set(np.unique(binarizar(cinza, 50))) <= {0, 255}


def test_limiar_fora_da_faixa():
    with pytest.raises(ValueError):
        binarizar(np.zeros((2, 2), dtype=np.uint8), 300)


def test_otsu_separa_dois_grupos_bem_definidos():
    rng = np.random.default_rng(1)
    escuros = rng.normal(60, 8, 5000)
    claros = rng.normal(190, 8, 5000)
    cinza = np.clip(np.concatenate([escuros, claros]), 0, 255).astype(np.uint8).reshape(100, 100)
    limiar = limiar_otsu(cinza)
    assert 100 < limiar < 150
    assert (binarizar(cinza) == 255).sum() == pytest.approx(5000, abs=50)


def test_otsu_em_imagem_de_uma_cor_so_nao_quebra():
    assert 0 <= limiar_otsu(np.full((5, 5), 77, dtype=np.uint8)) <= 255


def test_painel_coloca_lado_a_lado():
    a = np.zeros((10, 4, 3), dtype=np.uint8)
    b = np.zeros((10, 6), dtype=np.uint8)
    assert painel(a, b, margem=2).shape == (10, 12, 3)


def test_linha_de_comando_gera_os_tres_arquivos(tmp_path):
    from PIL import Image

    foto = tmp_path / "foto.png"
    Image.fromarray(np.random.default_rng(2).integers(0, 256, (16, 24, 3), dtype=np.uint8)).save(foto)
    assert main([str(foto), "--saida", str(tmp_path / "saida"), "--limiar", "100"]) == 0
    for sufixo in ("cinza", "binaria", "comparacao"):
        assert (tmp_path / "saida" / f"foto_{sufixo}.png").is_file()
    assert set(np.unique(carregar(tmp_path / "saida" / "foto_binaria.png"))) <= {0, 255}
