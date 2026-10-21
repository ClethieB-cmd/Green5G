"""Dados de entrada: simulador realista e leitor de CSV real.

Formato esperado (uma linha por célula por intervalo de medição):

    timestamp,celula_id,demanda,solar_w
    2026-08-03 00:00:00,CEL-01,14.2,0
    2026-08-03 00:15:00,CEL-01,13.8,0

- demanda: usuários ativos (ou outra medida de carga, ex. % de PRB) no intervalo
- solar_w: potência média do painel no intervalo (opcional; 0 se não houver painel)

Em dados reais isso costuma vir dos contadores de desempenho (PM counters)
exportados do OSS do fornecedor, normalmente a cada 15 minutos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Perfis de tráfego: (hora do pico, amplitude, largura em horas, multiplicador no fim de semana)
PERFIS = {
    "residencial": {"base": 12, "picos": [(8.0, 50, 2.0, 0.6), (13.0, 35, 1.7, 1.0), (20.5, 130, 2.3, 1.3)]},
    "comercial": {"base": 10, "picos": [(10.0, 110, 2.5, 0.3), (15.0, 100, 2.5, 0.3), (19.0, 40, 1.5, 0.6)]},
    "misto": {"base": 15, "picos": [(8.0, 60, 2.0, 0.7), (13.0, 45, 1.73, 1.0), (20.0, 120, 2.24, 1.3)]},
}


def gauss_circular(hora, centro, largura):
    """Pico gaussiano que dá a volta na meia-noite (23h fica a 1h de distância da 0h)."""
    d = np.abs(np.asarray(hora, dtype=float) - centro) % 24
    d = np.minimum(d, 24 - d)
    return np.exp(-(d ** 2) / (2 * largura ** 2))


def eh_fim_de_semana(tempos: pd.DatetimeIndex | pd.Series, feriados) -> np.ndarray:
    tempos = pd.DatetimeIndex(tempos)
    feriados = pd.to_datetime(list(feriados)).normalize()
    return np.asarray((tempos.dayofweek >= 5) | tempos.normalize().isin(feriados))


def _ar1(n, rng, phi, sigma, media=1.0, minimo=None, maximo=None):
    x = np.empty(n)
    x[0] = media
    for i in range(1, n):
        x[i] = media + phi * (x[i - 1] - media) + rng.normal(0, sigma)
    return np.clip(x, minimo, maximo)


def _simular_solar(tempos, rng, pico_w):
    hora = tempos.hour + tempos.minute / 60
    dia = (tempos.normalize() - tempos[0].normalize()).days
    nuvens = _ar1(dia.max() + 1, rng, phi=0.5, sigma=0.18, media=0.8, minimo=0.3, maximo=1.0)
    ceu_limpo = np.clip(np.sin((hora - 6) * np.pi / 12), 0, None) * ((hora >= 6) & (hora <= 18))
    ruido = np.clip(1 + rng.normal(0, 0.08, len(tempos)), 0.6, 1.1)
    return pico_w * ceu_limpo * nuvens[dia] * ruido


def simular_rede(dias: int = 70, freq_min: int = 15, n_celulas: int = 6, inicio: str = "2026-08-03",
                 feriados=(), solar_pico_w: float = 1000.0, semente: int = 42) -> pd.DataFrame:
    """Simula células de capacidade 5G sob uma macrocélula de cobertura.

    Inclui: perfis diferentes por célula, picos circulares, fim de semana e feriados,
    variação diária, tendência de crescimento, eventos inesperados (inclusive à noite),
    ruído, falhas de coleta e uma queda de 3 h por célula.
    """
    rng = np.random.default_rng(semente)
    passos_hora = 60 // freq_min
    tempos = pd.date_range(inicio, periods=dias * 24 * passos_hora, freq=f"{freq_min}min")
    hora = np.asarray(tempos.hour + tempos.minute / 60)
    dia_idx = np.asarray((tempos.normalize() - tempos[0].normalize()).days)
    fds = eh_fim_de_semana(tempos, feriados)
    solar = _simular_solar(tempos, rng, solar_pico_w)
    nomes_perfis = list(PERFIS)

    tabelas = []
    for c in range(n_celulas):
        nome_perfil = nomes_perfis[c % len(nomes_perfis)]
        perfil = PERFIS[nome_perfil]
        escala = rng.uniform(0.8, 1.2)
        desvio_h = rng.uniform(-0.5, 0.5)

        base = np.full(len(tempos), float(perfil["base"]))
        for centro, amp, largura, mult_fds in perfil["picos"]:
            base += amp * np.where(fds, mult_fds, 1.0) * gauss_circular(hora, centro + desvio_h, largura)

        nivel_diario = _ar1(dias, rng, phi=0.6, sigma=0.06)
        tendencia = 1 + 0.003 * dia_idx / 7
        demanda = base * escala * nivel_diario[dia_idx] * tendencia
        demanda = demanda * (1 + rng.normal(0, 0.10, len(demanda))) + rng.normal(0, 2.5, len(demanda))

        # Eventos inesperados (show, jogo, festa): 2 a 4 h com +30 a +90 usuários
        for _ in range(rng.poisson(dias / 14)):
            ini = rng.integers(0, len(tempos) - 5 * passos_hora)
            dur = rng.integers(2 * passos_hora, 4 * passos_hora + 1)
            forma = np.sin(np.linspace(0, np.pi, dur))
            demanda[ini:ini + dur] += rng.uniform(30, 90) * forma

        demanda = np.clip(demanda, 0, None)
        demanda[rng.random(len(demanda)) < 0.005] = np.nan          # falhas pontuais de coleta
        q = rng.integers(0, len(demanda) - 3 * passos_hora)
        demanda[q:q + 3 * passos_hora] = np.nan                     # queda de 3 h

        tabelas.append(pd.DataFrame({
            "timestamp": tempos,
            "celula_id": f"CEL-{c + 1:02d}",
            "perfil": nome_perfil,
            "demanda": np.round(demanda, 1),
            "solar_w": np.round(solar, 1),
        }))
    return pd.concat(tabelas, ignore_index=True)


# ----------------------------------------------------------------------------
# Leitura e limpeza de dados reais
# ----------------------------------------------------------------------------

def inferir_freq_min(df: pd.DataFrame) -> int:
    difs = (df.sort_values(["celula_id", "timestamp"])
              .groupby("celula_id")["timestamp"].diff().dropna())
    if difs.empty:
        raise ValueError("Dados insuficientes para inferir a frequência de medição.")
    return max(1, int(round(difs.median().total_seconds() / 60)))


def _interpolar_lacunas_curtas(s: pd.Series, max_passos: int) -> pd.Series:
    """Preenche só lacunas de até max_passos; lacunas maiores continuam vazias."""
    falta = s.isna()
    grupo = (falta != falta.shift()).cumsum()
    tamanho = falta.groupby(grupo).transform("sum")
    curta = falta & (tamanho <= max_passos)
    return s.where(~curta, s.interpolate(limit_area="inside"))


def regularizar(df: pd.DataFrame, max_lacuna_min: int = 60):
    """Coloca cada célula numa grade de tempo regular e trata lacunas.

    Retorna (df_limpo, freq_min, relatorio_qualidade).
    """
    freq = inferir_freq_min(df)
    max_passos = max(1, max_lacuna_min // freq)
    partes, relatorio = [], {}
    for cel, g in df.groupby("celula_id", sort=True):
        extras = {c: g[c].iloc[0] for c in g.columns if c not in ("timestamp", "celula_id", "demanda", "solar_w")}
        g = (g.set_index("timestamp")[["demanda", "solar_w"]]
               .resample(f"{freq}min").mean())
        faltando_antes = int(g["demanda"].isna().sum())
        g["demanda"] = _interpolar_lacunas_curtas(g["demanda"], max_passos)
        g["solar_w"] = g["solar_w"].fillna(0.0)
        faltando_depois = int(g["demanda"].isna().sum())
        g = g.reset_index()
        g["celula_id"] = cel
        for k, v in extras.items():
            g[k] = v
        partes.append(g)
        relatorio[cel] = {
            "pontos": len(g),
            "inicio": str(g["timestamp"].min()),
            "fim": str(g["timestamp"].max()),
            "faltando_pct": round(100 * faltando_antes / max(1, len(g)), 2),
            "preenchidos": faltando_antes - faltando_depois,
            "lacunas_longas_restantes": faltando_depois,
        }
    relatorio["_geral"] = {"freq_min": freq, "celulas": len(partes)}
    return pd.concat(partes, ignore_index=True), freq, relatorio


def carregar_csv(caminho: str, col_tempo: str = "timestamp", col_celula: str = "celula_id",
                 col_demanda: str = "demanda", col_solar: str = "solar_w", max_lacuna_min: int = 60):
    """Lê o CSV (simulado ou exportado do OSS), valida e regulariza."""
    df = pd.read_csv(caminho)
    df = df.rename(columns={col_tempo: "timestamp", col_celula: "celula_id",
                            col_demanda: "demanda", col_solar: "solar_w"})
    faltam = [c for c in ("timestamp", "celula_id", "demanda") if c not in df.columns]
    if faltam:
        raise ValueError(f"Colunas obrigatórias ausentes no CSV: {faltam}. "
                         f"Colunas encontradas: {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalidos = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])
    df["celula_id"] = df["celula_id"].astype(str)
    df["demanda"] = pd.to_numeric(df["demanda"], errors="coerce")
    df.loc[df["demanda"] < 0, "demanda"] = np.nan
    if "solar_w" not in df.columns:
        df["solar_w"] = 0.0
    df["solar_w"] = pd.to_numeric(df["solar_w"], errors="coerce").clip(lower=0)
    df = df.drop_duplicates(["celula_id", "timestamp"], keep="last")
    limpo, freq, relatorio = regularizar(df, max_lacuna_min)
    relatorio["_geral"]["timestamps_invalidos_descartados"] = invalidos
    return limpo, freq, relatorio
