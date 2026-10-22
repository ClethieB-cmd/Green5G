"""Painel web do Green5G (front-end).

Gera uma página HTML única, sem servidor, a partir dos resultados da avaliação.
Basta abrir o arquivo no navegador. Os gráficos usam a biblioteca Chart.js,
carregada pela internet; o restante funciona offline.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .politica import SLEEP

MODELO_HTML = Path(__file__).with_name("painel_template.html")
CHAVES = {"Sempre ativa": "ref", "Regras v1 (original)": "v1", "Green5G IA": "green",
          "Oráculo (limite teórico)": "oraculo"}


def _n(x, casas: int = 1):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, casas)


def _lista(s: pd.Series, casas: int = 1) -> list:
    return [_n(v, casas) for v in s.to_numpy(dtype=float)]


def dados_painel(res: dict, cfg, freq_min: int, perfis: dict | None = None, simulado: bool = False,
                 max_celulas: int = 50, max_decisoes: int = 3000) -> dict:
    green = res["_resultados"]["Green5G IA"]
    ref = res["_resultados"]["Sempre ativa"]
    perfis = perfis or {}

    politicas = []
    for nome, m in res["politicas"].items():
        politicas.append({
            "nome": nome, "chave": CHAVES.get(nome, nome),
            "energia_kwh": _n(m["energia_rede_kwh"]), "economia_pct": _n(m.get("economia_pct")),
            "sleep_pct": _n(m["tempo_sleep_pct"]), "violacoes_pct": _n(m["violacoes_qos_pct_tempo"], 2),
            "usuarios_h": _n(m["usuarios_hora_afetados"]), "religamentos": _n(m["religamentos_por_celula_dia"], 2),
        })

    ref_cel = ref.groupby("celula_id")["e_total_rede_wh"].sum()
    celulas, series = [], {}
    for cel in sorted(green["celula_id"].unique())[:max_celulas]:
        g = green[green["celula_id"] == cel].sort_values("timestamp")
        passos = g["timestamp"].diff().dropna().dt.total_seconds().div(60).unique()
        if len(passos) > 1 or (len(passos) == 1 and passos[0] != freq_min):
            raise ValueError(f"Série da célula {cel} não é regular; regularize os dados antes de gerar o painel.")
        series[cel] = {
            "t0": g["timestamp"].iloc[0].strftime("%Y-%m-%dT%H:%M"),
            "d": _lista(g["demanda"]), "p50": _lista(g["p50"]), "p90": _lista(g["p90"]),
            "m": "".join("S" if m == SLEEP else "A" for m in g["modo"]),
        }
        ult = g.iloc[-1]
        celulas.append({
            "id": cel, "perfil": perfis.get(cel),
            "economia_pct": _n(100 * (1 - g["e_total_rede_wh"].sum() / ref_cel[cel])),
            "sleep_pct": _n(100 * (g["modo"] == SLEEP).mean()),
            "violacoes": int(g["violacao_qos"].sum()),
            "ultimo": {"ts": ult["timestamp"].strftime("%Y-%m-%dT%H:%M"), "decisao": ult["decisao"],
                       "demanda": _n(ult["demanda"]), "p90": _n(ult["p90"]), "motivo": ult["motivo"]},
        })

    mud = green[green["decisao"] != green["modo"]].sort_values("timestamp", ascending=False).head(max_decisoes)
    decisoes = [{"ts": r.timestamp.strftime("%Y-%m-%dT%H:%M"), "cel": r.celula_id, "de": r.modo, "para": r.decisao,
                 "dem": _n(r.demanda), "p90": _n(r.p90), "motivo": r.motivo} for r in mud.itertuples()]

    hm = (green.assign(dia=green["timestamp"].dt.dayofweek, hora=green["timestamp"].dt.hour,
                       s=(green["modo"] == SLEEP))
               .pivot_table(index="dia", columns="hora", values="s", aggfunc="mean")
               .reindex(index=range(7), columns=range(24)) * 100)
    mapa = [[_n(v, 0) for v in linha] for linha in hm.to_numpy()]

    g_res = res["politicas"]["Green5G IA"]
    pg = res["previsao_geral"]
    cp, ce = cfg.politica, cfg.energia
    return {
        "meta": {
            "gerado_em": datetime.now().strftime("%Y-%m-%dT%H:%M"),
            "ini": green["timestamp"].min().strftime("%Y-%m-%dT%H:%M"),
            "fim": green["timestamp"].max().strftime("%Y-%m-%dT%H:%M"),
            "freq_min": int(freq_min), "n_celulas": int(green["celula_id"].nunique()),
            "simulado": bool(simulado), "semanas_teste": cfg.geral.semanas_teste,
            "capacidade": cp.capacidade_absorcao_macro, "margem_sleep": cp.margem_sleep,
            "margem_acordar": cp.margem_acordar, "horizonte_min": cp.horizonte_min,
            "tarifa": ce.tarifa_r_kwh, "fator_co2": ce.fator_co2_kg_kwh,
        },
        "previsao": {"mae": _n(pg["mae_p50"]), "cobertura": _n(pg["cobertura_p90_pct"]),
                     "melhoria": _n(pg["melhoria_vs_ingenuo_pct"], 0)},
        "projecao": {"kwh_ano": _n(g_res["economia_kwh_celula_ano"], 0), "r_ano": _n(g_res["economia_r_celula_ano"], 0),
                     "co2_ano": _n(g_res["co2_evitado_kg_celula_ano"])},
        "politicas": politicas,
        "sensibilidade": [{"margem": s["margem_sleep"], "economia_pct": _n(s["economia_pct"]),
                           "usuarios_h": _n(s["usuarios_hora_afetados"])} for s in res["sensibilidade"]],
        "celulas": celulas, "series": series, "mapa": mapa, "decisoes": decisoes,
        "total_decisoes": int(len(green[green["decisao"] != green["modo"]])),
    }


def gerar_painel(res: dict, cfg, freq_min: int, caminho: str, perfis: dict | None = None,
                 simulado: bool = False, documento_completo: bool = True) -> str:
    """Escreve o painel. documento_completo=False gera só o conteúdo (para publicar como página hospedada)."""
    dados = dados_painel(res, cfg, freq_min, perfis, simulado)
    js = json.dumps(dados, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    cabeca, corpo = MODELO_HTML.read_text(encoding="utf-8").split("<!--CORPO-->")
    corpo = corpo.replace("/*__DADOS__*/null", js)
    if documento_completo:
        html = ("<!doctype html>\n<html lang=\"pt-BR\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                f"{cabeca}</head>\n<body>\n{corpo}</body>\n</html>\n")
    else:
        html = cabeca + corpo
    Path(caminho).write_text(html, encoding="utf-8")
    return caminho
