"""Previsor do pico de demanda da próxima janela, com incerteza calibrada.

- Gradient boosting com perda quantílica: prevê a mediana (P50) e um limite
  alto (P90) do pico de demanda da próxima hora.
- Calibração conformal: ajusta o P90 com a última semana de treino para que
  ele realmente cubra ~90% dos casos. É isso que torna a decisão de dormir
  segura: a célula só dorme se até o cenário pessimista couber na macro.
- Um único modelo global para todas as células: funciona em célula nova sem
  retreinar, porque aprende a partir do histórico recente da própria célula.
"""
from __future__ import annotations

import math
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import ConfigModelo
from .features import FEATURES

VERSAO_MODELO = "2.0"


class PrevisorDemanda:
    def __init__(self, cfg: ConfigModelo, freq_min: int | None = None, horizonte_min: int | None = None):
        self.cfg = cfg
        self.freq_min = freq_min
        self.horizonte_min = horizonte_min
        self.modelos: dict[str, HistGradientBoostingRegressor] = {}
        self.ajuste_conformal = 0.0
        self.meta: dict = {}

    def _novo_modelo(self, quantil: float) -> HistGradientBoostingRegressor:
        c = self.cfg
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=quantil, max_iter=c.max_iter, learning_rate=c.learning_rate,
            max_leaf_nodes=c.max_leaf_nodes, min_samples_leaf=c.min_samples_leaf,
            l2_regularization=c.l2_regularization, early_stopping=False, random_state=c.semente,
        )

    def _prever_bruto(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.modelos["p50"].predict(X), self.modelos["p_seg"].predict(X)

    def treinar(self, feats: pd.DataFrame) -> "PrevisorDemanda":
        dados = feats.dropna(subset=["alvo"])
        corte = dados["timestamp"].max() - pd.Timedelta(days=self.cfg.dias_calibracao)
        # Purga: no treino só entram linhas cujo alvo terminou antes do corte.
        treino = dados[dados["alvo_fim"] <= corte]
        calib = dados[dados["timestamp"] > corte]
        if len(treino) < 1000 or len(calib) < 100:
            raise ValueError(f"Histórico insuficiente para treinar (treino={len(treino)}, calibração={len(calib)}). "
                             "Use pelo menos ~3 semanas de dados.")

        self.modelos["p50"] = self._novo_modelo(0.5).fit(treino[FEATURES], treino["alvo"])
        self.modelos["p_seg"] = self._novo_modelo(self.cfg.quantil_seguranca).fit(treino[FEATURES], treino["alvo"])

        p50_cal, pseg_cal = self._prever_bruto(calib[FEATURES])
        erros = calib["alvo"].to_numpy() - pseg_cal
        n = len(erros)
        nivel = min(1.0, math.ceil((n + 1) * self.cfg.quantil_seguranca) / n)
        self.ajuste_conformal = float(np.quantile(erros, nivel))

        self.meta = {
            "versao": VERSAO_MODELO,
            "treinado_em": datetime.now().isoformat(timespec="seconds"),
            "periodo_treino": [str(treino["timestamp"].min()), str(treino["timestamp"].max())],
            "periodo_calibracao": [str(calib["timestamp"].min()), str(calib["timestamp"].max())],
            "amostras_treino": int(len(treino)),
            "amostras_calibracao": int(n),
            "celulas": sorted(dados["celula_id"].unique().tolist()),
            "cobertura_p_seg_antes_calibracao": float(np.mean(calib["alvo"].to_numpy() <= pseg_cal)),
            "ajuste_conformal": self.ajuste_conformal,
            "mae_p50_calibracao": float(np.mean(np.abs(calib["alvo"].to_numpy() - p50_cal))),
        }
        return self

    def prever(self, feats: pd.DataFrame) -> pd.DataFrame:
        if not self.modelos:
            raise RuntimeError("Modelo não treinado.")
        p50, pseg = self._prever_bruto(feats[FEATURES])
        p50 = np.clip(p50, 0, None)
        pseg = np.maximum(pseg + self.ajuste_conformal, p50)
        return pd.DataFrame({"p50": p50, "p90": pseg}, index=feats.index)

    # ------------------------------------------------------------------ persistência
    def salvar(self, caminho: str) -> None:
        joblib.dump({
            "versao": VERSAO_MODELO, "cfg": self.cfg, "freq_min": self.freq_min,
            "horizonte_min": self.horizonte_min, "features": FEATURES, "modelos": self.modelos,
            "ajuste_conformal": self.ajuste_conformal, "meta": self.meta,
        }, caminho)

    @classmethod
    def carregar(cls, caminho: str) -> "PrevisorDemanda":
        pacote = joblib.load(caminho)
        if pacote.get("features") != FEATURES:
            raise ValueError("O modelo salvo usa outras features. Retreine com esta versão do código.")
        obj = cls(pacote["cfg"], pacote["freq_min"], pacote["horizonte_min"])
        obj.modelos = pacote["modelos"]
        obj.ajuste_conformal = pacote["ajuste_conformal"]
        obj.meta = pacote["meta"]
        return obj
