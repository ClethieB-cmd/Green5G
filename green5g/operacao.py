"""Uso em operação: decidir o próximo intervalo de cada célula a partir do histórico recente.

Pensado para rodar a cada 15 min (cron, agendador do OSS, rApp etc.):
  1. lê o histórico recente (pelo menos 8 dias, para as features de semana passada)
  2. prevê o pico da próxima hora com o modelo salvo
  3. aplica a política com o estado atual de cada célula (estado.json)
  4. devolve a decisão + motivo e grava um log de auditoria

Em modo sombra, a operadora só registra as decisões, sem aplicá-las, e depois
compara com o que aconteceu de verdade antes de ligar o controle real.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .config import Config
from .features import construir_features, passos
from .modelo import PrevisorDemanda
from .politica import ATIVA, Situacao, decidir_green5g


def _ler_estado(caminho: str | None) -> dict:
    if caminho and os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _minutos_no_modo(dados_ate: pd.Timestamp, desde: str | None, freq_min: int) -> float:
    """Minutos no modo atual ao fim do último intervalo com dados.

    `desde` é o início do primeiro intervalo no modo atual, e `dados_ate` é o início
    do último intervalo medido. Esse último intervalo já foi vivido no modo atual,
    por isso entra na conta (+ freq_min), igual ao backtest em avaliacao.py.
    """
    if not desde:
        return 1e9
    return (dados_ate + pd.Timedelta(minutes=freq_min) - pd.Timestamp(desde)).total_seconds() / 60


def decidir_agora(df: pd.DataFrame, freq_min: int, modelo: PrevisorDemanda, cfg: Config,
                  caminho_estado: str | None = None, caminho_log: str | None = None) -> list[dict]:
    if modelo.freq_min != freq_min:
        raise ValueError(f"O modelo foi treinado com medições a cada {modelo.freq_min} min, "
                         f"mas os dados chegaram a cada {freq_min} min.")
    cp = cfg.politica
    feats = construir_features(df, freq_min, modelo.horizonte_min, cfg.geral.feriados)
    feats[["p50", "p90"]] = modelo.prever(feats)
    fim_calib = pd.Timestamp(modelo.meta["periodo_calibracao"][1])
    janela = passos(freq_min, cp.janela_monitor_h * 60)
    estado = _ler_estado(caminho_estado)

    decisoes = []
    for cel, g in feats.groupby("celula_id", sort=True):
        g = g.sort_values("timestamp")
        ult = g.iloc[-1]
        # Monitor de calibração só com previsões fora da amostra (depois do treino).
        hist = g[(g["timestamp"] > fim_calib)].dropna(subset=["alvo"]).tail(janela)
        cobertura = float(np.mean(hist["alvo"] <= hist["p90"])) if len(hist) >= cp.amostras_min_monitor else None

        est = estado.get(cel, {"modo": ATIVA, "desde": None})
        minutos = _minutos_no_modo(ult["timestamp"], est["desde"], freq_min)
        s = Situacao(hora=int(ult["timestamp"].hour), demanda_atual=float(ult["demanda"]),
                     p50=float(ult["p50"]), p90=float(ult["p90"]), solar_w=float(ult["solar_w"]),
                     modo_atual=est["modo"], minutos_no_modo=minutos, cobertura_recente=cobertura)
        novo, motivo = decidir_green5g(s, cp)

        inicio = ult["timestamp"] + pd.Timedelta(minutes=freq_min)
        if novo != est["modo"]:
            estado[cel] = {"modo": novo, "desde": str(inicio)}
        else:
            estado.setdefault(cel, est)
        decisoes.append({
            "celula_id": cel,
            "dados_ate": str(ult["timestamp"]),
            "valido_a_partir_de": str(inicio),
            "modo": novo,
            "mudou": novo != est["modo"],
            "demanda_atual": None if np.isnan(s.demanda_atual) else round(s.demanda_atual, 1),
            "pico_previsto_p50": round(s.p50, 1),
            "pico_previsto_p90": round(s.p90, 1),
            "cobertura_recente": None if cobertura is None else round(cobertura, 3),
            "motivo": motivo,
        })

    if caminho_estado:
        with open(caminho_estado, "w", encoding="utf-8") as f:
            json.dump(estado, f, ensure_ascii=False, indent=2)
    if caminho_log:
        pd.DataFrame(decisoes).to_csv(caminho_log, mode="a", index=False,
                                      header=not os.path.exists(caminho_log))
    return decisoes
