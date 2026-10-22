"""Testes automáticos. Rodar com:  python -m pytest -q"""
import math

import numpy as np
import pandas as pd
import pytest

from green5g.config import Config
from green5g.dados import carregar_csv, gauss_circular, simular_rede
from green5g.energia import contabilizar
from green5g.features import FEATURES, construir_features, max_futuro
from green5g.operacao import _minutos_no_modo
from green5g.politica import ATIVA, SLEEP, Situacao, decidir_green5g


@pytest.fixture(scope="module")
def cfg():
    return Config()


@pytest.fixture(scope="module")
def dados(cfg):
    return simular_rede(dias=28, n_celulas=2, feriados=cfg.geral.feriados, semente=1)


def test_pico_circular_sem_salto_na_meia_noite():
    # Sem salto: 23h59 e 0h00 ficam praticamente no mesmo ponto da curva.
    assert gauss_circular(23.99, 20, 2) == pytest.approx(gauss_circular(0.0, 20, 2), rel=0.02)
    # Dá a volta: com pico às 23h, 22h e 0h ficam a 1 h do pico e valem o mesmo.
    assert gauss_circular(22, 23, 2) == pytest.approx(gauss_circular(0, 23, 2))
    assert gauss_circular(0, 20, 2.24) > 0.1  # no código original isso dava ~0


def test_max_futuro():
    s = pd.Series([1, 5, 2, 8, 3, 0], dtype=float)
    r = max_futuro(s, 2)
    assert r.iloc[0] == 5 and r.iloc[1] == 8 and r.iloc[2] == 8 and r.iloc[3] == 3
    assert math.isnan(r.iloc[5])


def test_features_nao_usam_o_futuro(dados, cfg):
    """Mudar a demanda depois de t não pode mudar nenhuma feature em t."""
    base = construir_features(dados, 15, 60, cfg.geral.feriados)
    t = dados["timestamp"].iloc[len(dados) // 4]
    alterado = dados.copy()
    alterado.loc[alterado["timestamp"] > t, "demanda"] *= 3
    novo = construir_features(alterado, 15, 60, cfg.geral.feriados)
    linha_a = base[base["timestamp"] == t][FEATURES].to_numpy()
    linha_b = novo[novo["timestamp"] == t][FEATURES].to_numpy()
    np.testing.assert_allclose(linha_a, linha_b, equal_nan=True)


def test_modo_seguro_sem_dados(cfg):
    s = Situacao(hora=3, demanda_atual=float("nan"), p90=10.0, modo_atual=SLEEP)
    modo, motivo = decidir_green5g(s, cfg.politica)
    assert modo == ATIVA and "seguro" in motivo


def test_modo_seguro_previsao_descalibrada(cfg):
    s = Situacao(hora=3, demanda_atual=5.0, p90=10.0, cobertura_recente=0.5)
    assert decidir_green5g(s, cfg.politica)[0] == ATIVA


def test_protecao_reativa_acorda_na_hora(cfg):
    s = Situacao(hora=3, demanda_atual=cfg.politica.capacidade_absorcao_macro + 1, p90=5.0, modo_atual=SLEEP)
    assert decidir_green5g(s, cfg.politica)[0] == ATIVA


def test_dorme_com_baixa_demanda_e_respeita_permanencia(cfg):
    s = Situacao(hora=3, demanda_atual=5.0, p90=10.0, modo_atual=ATIVA, minutos_no_modo=120)
    assert decidir_green5g(s, cfg.politica)[0] == SLEEP
    s.minutos_no_modo = 15
    assert decidir_green5g(s, cfg.politica)[0] == ATIVA


def test_energia_sempre_ativa(cfg):
    ce, cp = cfg.energia, cfg.politica
    r = pd.DataFrame({"celula_id": ["A"] * 4, "demanda": [0.0] * 4, "solar_w": [0.0] * 4, "modo": [ATIVA] * 4})
    out = contabilizar(r, ce, cp, 15)
    assert out["e_celula_wh"].sum() == pytest.approx(ce.potencia_ociosa_w * 1.0)  # 4 x 15 min = 1 h


def test_violacao_qos_contada(cfg):
    ce, cp = cfg.energia, cfg.politica
    r = pd.DataFrame({"celula_id": ["A"] * 2, "demanda": [10.0, cp.capacidade_absorcao_macro + 20],
                      "solar_w": [0.0, 0.0], "modo": [SLEEP, SLEEP]})
    out = contabilizar(r, ce, cp, 15)
    assert out["violacao_qos"].tolist() == [False, True]
    assert out["usuarios_hora_afetados"].sum() == pytest.approx(20 * 0.25)


def test_painel_web_gerado(tmp_path, dados):
    from green5g.avaliacao import backtest
    from green5g.dados import regularizar
    from green5g.painel import gerar_painel
    cfg = Config()
    cfg.geral.semanas_teste = 1
    df, freq, _ = regularizar(dados)
    res = backtest(df, freq, cfg, verbose=False)
    html = open(gerar_painel(res, cfg, freq, str(tmp_path / "painel.html"), simulado=True), encoding="utf-8").read()
    assert html.startswith("<!doctype html>") and "<title>Painel Green5G</title>" in html
    assert "__DADOS__" not in html and '"simulado":true' in html


def test_leitor_csv_valida_colunas(tmp_path):
    caminho = tmp_path / "ruim.csv"
    pd.DataFrame({"tempo": ["2026-01-01"], "x": [1]}).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="Colunas obrigatórias"):
        carregar_csv(str(caminho))


def test_leitor_csv_preenche_so_lacunas_curtas(tmp_path, dados):
    caminho = tmp_path / "ok.csv"
    dados.to_csv(caminho, index=False)
    df, freq, rel = carregar_csv(str(caminho))
    assert freq == 15
    assert rel["CEL-01"]["lacunas_longas_restantes"] >= 12  # a queda de 3 h continua vazia


def test_minutos_no_modo_igual_ao_backtest():
    """Na operação, o tempo no modo conta o último intervalo, como no backtest."""
    desde = "2026-10-26 00:15:00"             # célula entrou no modo neste intervalo
    ult = pd.Timestamp("2026-10-26 00:15:00")  # e esse intervalo acabou de ser medido
    assert _minutos_no_modo(ult, desde, 15) == 15
    assert _minutos_no_modo(ult + pd.Timedelta(minutes=45), desde, 15) == 60
    assert _minutos_no_modo(ult, None, 15) == 1e9
