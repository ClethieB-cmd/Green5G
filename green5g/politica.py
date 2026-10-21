"""Políticas de decisão (ATIVA / SLEEP) para cada célula de capacidade.

Ordem de prioridade da política Green5G (QoS sempre antes de economia):
  1. Dados ausentes                      -> ATIVA (modo seguro)
  2. Previsão descalibrada (drift)       -> ATIVA (modo seguro)
  3. Hora protegida pela operadora       -> ATIVA
  4. Em SLEEP: demanda observada acima da capacidade da macro -> acorda já
               P90 do pico da próxima hora acima do limiar     -> acorda
  5. Em ATIVA: dorme só se P90 E demanda atual estão abaixo do limiar de sleep
               e a célula está ativa há o tempo mínimo (anti-oscilação)

Cada decisão volta com um motivo legível, para auditoria pela operadora.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ConfigPolitica

ATIVA, SLEEP = "ATIVA", "SLEEP"


@dataclass
class Situacao:
    """O que a política sabe no fim do intervalo atual."""
    hora: int
    demanda_atual: float
    p50: float = math.nan
    p90: float = math.nan
    alvo_real: float = math.nan      # só o oráculo usa
    solar_w: float = 0.0
    modo_atual: str = ATIVA
    minutos_no_modo: float = 1e9
    cobertura_recente: float | None = None


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def decidir_green5g(s: Situacao, cfg: ConfigPolitica) -> tuple[str, str]:
    cap = cfg.capacidade_absorcao_macro
    lim_sleep, lim_acordar = cfg.margem_sleep * cap, cfg.margem_acordar * cap

    if _nan(s.demanda_atual) or _nan(s.p90):
        return ATIVA, "modo seguro: dado de demanda ou previsão ausente"
    if s.cobertura_recente is not None and s.cobertura_recente < cfg.cobertura_minima:
        return ATIVA, f"modo seguro: previsão descalibrada (P90 acertou {s.cobertura_recente:.0%}); retreinar"
    if s.hora in cfg.horas_protegidas:
        return ATIVA, f"hora {s.hora}h protegida pela operadora"

    if s.modo_atual == SLEEP:
        if s.demanda_atual > cap:
            return ATIVA, f"proteção reativa: demanda {s.demanda_atual:.0f} > capacidade da macro {cap:.0f}"
        if s.p90 > lim_acordar:
            return ATIVA, f"previsão: pico P90 {s.p90:.1f} > limiar de religar {lim_acordar:.1f}"
        return SLEEP, f"mantém sleep: pico P90 {s.p90:.1f} <= {lim_acordar:.1f}"

    if s.p90 < lim_sleep and s.demanda_atual < lim_sleep:
        if s.minutos_no_modo < cfg.permanencia_min_ativa_min:
            return ATIVA, f"poderia dormir, mas está ativa há só {s.minutos_no_modo:.0f} min"
        return SLEEP, f"dorme: pico P90 da próxima hora {s.p90:.1f} < limiar {lim_sleep:.1f}"
    return ATIVA, f"mantém ativa: pico P90 {s.p90:.1f} >= limiar {lim_sleep:.1f}"


def decidir_sempre_ativa(s: Situacao, cfg: ConfigPolitica) -> tuple[str, str]:
    return ATIVA, "referência: sempre ligada"


def decidir_regras_v1(s: Situacao, cfg: ConfigPolitica) -> tuple[str, str]:
    """Sistema de regras do projeto original (reativo, sem previsão)."""
    if _nan(s.demanda_atual):
        return s.modo_atual, "regras v1: sem dado, mantém"
    if s.hora in (7, 8, 12, 13, 18, 19, 20, 21):
        return ATIVA, "regras v1: horário de pico"
    if 0 <= s.hora <= 5 and s.demanda_atual < 20:
        return SLEEP, "regras v1: madrugada"
    modo = s.modo_atual
    if modo == SLEEP and s.demanda_atual > 40:
        modo = ATIVA
    elif modo == ATIVA and s.demanda_atual < 25:
        modo = SLEEP
    if s.solar_w > 500 and s.demanda_atual > 15:
        modo = ATIVA
    return modo, "regras v1"


def decidir_oraculo(s: Situacao, cfg: ConfigPolitica) -> tuple[str, str]:
    """Limite teórico: conhece o pico real da próxima hora. Não existe na prática."""
    if _nan(s.alvo_real):
        return ATIVA, "oráculo: sem dado"
    return (SLEEP, "oráculo") if s.alvo_real <= cfg.capacidade_absorcao_macro else (ATIVA, "oráculo")


POLITICAS = {
    "Sempre ativa": decidir_sempre_ativa,
    "Regras v1 (original)": decidir_regras_v1,
    "Green5G IA": decidir_green5g,
    "Oráculo (limite teórico)": decidir_oraculo,
}
