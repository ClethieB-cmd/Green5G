"""Contabilidade de energia, custo, CO2 e qualidade de serviço (QoS).

A contabilidade é a mesma para todas as políticas, e todas têm o mesmo
painel solar. Assim a economia medida é só a da decisão de dormir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ConfigEnergia, ConfigPolitica
from .politica import ATIVA, SLEEP


def contabilizar(res: pd.DataFrame, ce: ConfigEnergia, cp: ConfigPolitica, freq_min: int) -> pd.DataFrame:
    """Recebe por passo: celula_id, demanda, solar_w, modo. Devolve as colunas de energia e QoS."""
    r = res.copy()
    dt_h = freq_min / 60
    dem = r.groupby("celula_id")["demanda"].transform(lambda s: s.ffill().bfill()).fillna(0)
    dormindo = (r["modo"] == SLEEP).to_numpy()

    carga = np.clip(dem / ce.capacidade_celula_usuarios, 0, 1)
    p_ativa = ce.potencia_ociosa_w + (ce.potencia_max_w - ce.potencia_ociosa_w) * carga
    p_cel = np.where(dormindo, ce.potencia_sleep_w, p_ativa)

    anterior = r.groupby("celula_id")["modo"].shift(1)
    religou = ((anterior == SLEEP) & (r["modo"] == ATIVA)).to_numpy()

    cap = cp.capacidade_absorcao_macro
    absorvidos = np.where(dormindo, np.minimum(dem, cap), 0.0)
    nao_atendidos = np.where(dormindo, np.clip(dem - cap, 0, None), 0.0)

    r["e_celula_wh"] = p_cel * dt_h + religou * ce.energia_religamento_wh
    r["e_solar_wh"] = r["solar_w"].fillna(0).to_numpy() * ce.eficiencia_solar * dt_h
    r["e_rede_celula_wh"] = np.clip(r["e_celula_wh"] - r["e_solar_wh"], 0, None)
    r["e_macro_extra_wh"] = absorvidos * ce.custo_macro_w_por_usuario * dt_h
    r["e_total_rede_wh"] = r["e_rede_celula_wh"] + r["e_macro_extra_wh"]
    r["religou"] = religou
    r["violacao_qos"] = dormindo & (dem.to_numpy() > cap)
    r["usuarios_hora_afetados"] = nao_atendidos * dt_h
    return r


def resumir(r: pd.DataFrame, ce: ConfigEnergia, freq_min: int, referencia: pd.DataFrame | None = None) -> dict:
    n_cel = r["celula_id"].nunique()
    dias = len(r) * freq_min / (60 * 24) / n_cel
    total = r["e_total_rede_wh"].sum() / 1000
    sem_solar = (r["e_celula_wh"].sum() + r["e_macro_extra_wh"].sum()) / 1000
    out = {
        "energia_rede_kwh": total,
        "energia_sem_solar_kwh": sem_solar,
        "energia_macro_extra_kwh": r["e_macro_extra_wh"].sum() / 1000,
        "tempo_sleep_pct": 100 * (r["modo"] == SLEEP).mean(),
        "violacoes_qos_pct_tempo": 100 * r["violacao_qos"].mean(),
        "usuarios_hora_afetados": r["usuarios_hora_afetados"].sum(),
        "religamentos_por_celula_dia": r["religou"].sum() / (n_cel * dias),
        "dias": dias,
        "celulas": n_cel,
    }
    if referencia is not None:
        ref_total = referencia["e_total_rede_wh"].sum() / 1000
        ref_sem_solar = (referencia["e_celula_wh"].sum() + referencia["e_macro_extra_wh"].sum()) / 1000
        economia = ref_total - total
        por_cel_ano = economia / (n_cel * dias) * 365
        out.update({
            "economia_kwh": economia,
            "economia_pct": 100 * economia / ref_total,
            "economia_pct_sem_solar": 100 * (ref_sem_solar - sem_solar) / ref_sem_solar,
            "economia_kwh_celula_ano": por_cel_ano,
            "economia_r_celula_ano": por_cel_ano * ce.tarifa_r_kwh,
            "co2_evitado_kg_celula_ano": por_cel_ano * ce.fator_co2_kg_kwh,
        })
    return out
