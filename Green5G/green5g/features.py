"""Engenharia de features sem vazamento de dados.

Regra de ouro: as features do instante t usam SÓ dados até t.
O alvo é o pico de demanda da próxima janela (t+1 ... t+H), que é
exatamente o que a decisão de dormir ou não precisa saber.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dados import eh_fim_de_semana

FEATURES = [
    "hora_sin", "hora_cos", "dia_sem_sin", "dia_sem_cos", "fim_de_semana", "feriado",
    "demanda_atual", "lag_1", "lag_1h", "media_1h", "max_3h", "tendencia_1h",
    "demanda_ontem", "max_janela_ontem", "max_janela_semana", "media_janela_7d",
]


def passos(freq_min: int, minutos: int) -> int:
    return max(1, int(round(minutos / freq_min)))


def max_futuro(s: pd.Series, h: int) -> pd.Series:
    """Máximo de s em t+1 ... t+h (NaN se faltar algum ponto)."""
    return s.shift(-1).rolling(h, min_periods=h).max().shift(-(h - 1))


def construir_features(df: pd.DataFrame, freq_min: int, horizonte_min: int, feriados) -> pd.DataFrame:
    H = passos(freq_min, horizonte_min)
    dia = passos(freq_min, 24 * 60)
    semana = 7 * dia
    if H >= dia:
        raise ValueError("O horizonte de decisão precisa ser menor que 24 h.")
    feriados_idx = pd.to_datetime(list(feriados)).normalize()

    blocos = []
    for cel, g in df.groupby("celula_id", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        ts = pd.DatetimeIndex(g["timestamp"])
        d = g["demanda"].astype(float)
        hora = np.asarray(ts.hour + ts.minute / 60)
        dow = np.asarray(ts.dayofweek)
        alvo = max_futuro(d, H)

        f = pd.DataFrame({
            "timestamp": g["timestamp"].values,
            "celula_id": cel,
            "demanda": d.values,
            "solar_w": g["solar_w"].values,
            "hora_sin": np.sin(2 * np.pi * hora / 24),
            "hora_cos": np.cos(2 * np.pi * hora / 24),
            "dia_sem_sin": np.sin(2 * np.pi * dow / 7),
            "dia_sem_cos": np.cos(2 * np.pi * dow / 7),
            "fim_de_semana": eh_fim_de_semana(ts, feriados).astype(int),
            "feriado": np.asarray(ts.normalize().isin(feriados_idx)).astype(int),
            "demanda_atual": d.values,
            "lag_1": d.shift(1).values,
            "lag_1h": d.shift(passos(freq_min, 60)).values,
            "media_1h": d.rolling(passos(freq_min, 60), min_periods=1).mean().values,
            "max_3h": d.rolling(passos(freq_min, 180), min_periods=1).max().values,
            "tendencia_1h": (d - d.shift(passos(freq_min, 60))).values,
            "demanda_ontem": d.shift(dia).values,
            # Pico da MESMA janela futura ontem e na semana passada: tudo já aconteceu em t.
            "max_janela_ontem": alvo.shift(dia).values,
            "max_janela_semana": alvo.shift(semana).values,
            "media_janela_7d": pd.concat([alvo.shift(k * dia) for k in range(1, 8)], axis=1).mean(axis=1).values,
            "alvo": alvo.values,
        })
        f["alvo_fim"] = f["timestamp"] + pd.Timedelta(minutes=H * freq_min)
        blocos.append(f)
    return pd.concat(blocos, ignore_index=True)
