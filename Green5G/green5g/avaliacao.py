"""Backtest walk-forward: simula o uso real, semana a semana.

Para cada semana de teste, o modelo é treinado só com o passado (com purga
na fronteira) e decide passo a passo, sem ver o futuro. Todas as políticas
rodam sobre os mesmos dados, com a mesma contabilidade de energia.
"""
from __future__ import annotations

import copy
from collections import deque

import numpy as np
import pandas as pd

from .config import Config
from .energia import contabilizar, resumir
from .features import construir_features, passos
from .modelo import PrevisorDemanda
from .politica import ATIVA, POLITICAS, Situacao


def executar_politica(serie: pd.DataFrame, decisor, cfg: Config, freq_min: int) -> pd.DataFrame:
    """Roda uma política numa célula. A decisão tomada no fim do passo i vale para o passo i+1."""
    cp = cfg.politica
    H = passos(freq_min, cp.horizonte_min)
    janela = deque(maxlen=passos(freq_min, cp.janela_monitor_h * 60))

    hora = pd.DatetimeIndex(serie["timestamp"]).hour.to_numpy()
    dem = serie["demanda"].to_numpy(float)
    p50 = serie["p50"].to_numpy(float)
    p90 = serie["p90"].to_numpy(float)
    alvo = serie["alvo"].to_numpy(float)
    solar = serie["solar_w"].to_numpy(float)

    n = len(serie)
    modos = np.empty(n, dtype=object)       # estado da célula durante o passo i
    decisoes = np.empty(n, dtype=object)    # decisão tomada no fim do passo i (vale para i+1)
    motivos = np.empty(n, dtype=object)     # motivo dessa decisão
    modos[0] = ATIVA
    minutos = 1e9
    for i in range(n):
        # O acerto da previsão feita em i-H só é conhecido agora, em i.
        j = i - H
        if j >= 0 and not np.isnan(alvo[j]) and not np.isnan(p90[j]):
            janela.append(alvo[j] <= p90[j])
        cobertura = (sum(janela) / len(janela)) if len(janela) >= cp.amostras_min_monitor else None
        s = Situacao(hora=int(hora[i]), demanda_atual=dem[i], p50=p50[i], p90=p90[i], alvo_real=alvo[i],
                     solar_w=solar[i], modo_atual=modos[i], minutos_no_modo=minutos,
                     cobertura_recente=cobertura)
        novo, motivo = decisor(s, cp)
        decisoes[i], motivos[i] = novo, motivo
        if i + 1 < n:
            minutos = minutos + freq_min if novo == modos[i] else freq_min
            modos[i + 1] = novo
    out = serie[["timestamp", "celula_id", "demanda", "solar_w", "p50", "p90", "alvo"]].copy()
    out["modo"] = modos
    out["decisao"] = decisoes
    out["motivo"] = motivos
    return out


def metricas_previsao(pred: pd.DataFrame) -> dict:
    v = pred.dropna(subset=["alvo"])
    ingenuo = v.dropna(subset=["max_janela_semana"])
    mae_ia = float(np.mean(np.abs(v["alvo"] - v["p50"])))
    mae_ia_mesmas = float(np.mean(np.abs(ingenuo["alvo"] - ingenuo["p50"])))
    mae_ing = float(np.mean(np.abs(ingenuo["alvo"] - ingenuo["max_janela_semana"])))
    return {
        "mae_p50": mae_ia,
        "mae_ingenuo_semana_passada": mae_ing,
        "melhoria_vs_ingenuo_pct": 100 * (1 - mae_ia_mesmas / mae_ing),
        "cobertura_p90_pct": 100 * float(np.mean(v["alvo"] <= v["p90"])),
        "largura_media_p50_p90": float(np.mean(v["p90"] - v["p50"])),
    }


def backtest(df: pd.DataFrame, freq_min: int, cfg: Config, verbose: bool = True):
    feats = construir_features(df, freq_min, cfg.politica.horizonte_min, cfg.geral.feriados)
    passo = pd.Timedelta(minutes=freq_min)
    inicio_teste = (feats["timestamp"].max() + passo).normalize() - pd.Timedelta(weeks=cfg.geral.semanas_teste)

    blocos, folds = [], []
    for k in range(cfg.geral.semanas_teste):
        ini = inicio_teste + pd.Timedelta(weeks=k)
        fim = ini + pd.Timedelta(weeks=1)
        treino = feats[feats["alvo_fim"] < ini]
        teste = feats[(feats["timestamp"] >= ini) & (feats["timestamp"] < fim)].copy()
        modelo = PrevisorDemanda(cfg.modelo, freq_min, cfg.politica.horizonte_min).treinar(treino)
        teste[["p50", "p90"]] = modelo.prever(teste)
        m = metricas_previsao(teste)
        m.update({"semana": k + 1, "inicio": str(ini.date()), "treino_ate": modelo.meta["periodo_treino"][1],
                  "ajuste_conformal": modelo.ajuste_conformal})
        folds.append(m)
        blocos.append(teste)
        if verbose:
            print(f"  semana {k + 1}: MAE {m['mae_p50']:.1f} usuários | cobertura P90 {m['cobertura_p90_pct']:.1f}% "
                  f"| {m['melhoria_vs_ingenuo_pct']:.0f}% melhor que repetir a semana passada")
    pred = pd.concat(blocos, ignore_index=True).sort_values(["celula_id", "timestamp"])

    resultados = {}
    for nome, decisor in POLITICAS.items():
        partes = [executar_politica(g.reset_index(drop=True), decisor, cfg, freq_min)
                  for _, g in pred.groupby("celula_id", sort=True)]
        resultados[nome] = contabilizar(pd.concat(partes, ignore_index=True), cfg.energia, cfg.politica, freq_min)

    ref = resultados["Sempre ativa"]
    resumo = {nome: resumir(r, cfg.energia, freq_min, None if nome == "Sempre ativa" else ref)
              for nome, r in resultados.items()}
    return {
        "previsao_por_semana": folds,
        "previsao_geral": metricas_previsao(pred),
        "politicas": resumo,
        "sensibilidade": sensibilidade(pred, ref, cfg, freq_min),
        "_pred": pred,
        "_resultados": resultados,
    }


def sensibilidade(pred: pd.DataFrame, ref: pd.DataFrame, cfg: Config, freq_min: int,
                  margens=(0.70, 0.80, 0.85, 0.90, 1.00)) -> list[dict]:
    """Curva economia x risco: como os resultados mudam com a margem de sleep."""
    linhas = []
    for m in margens:
        c = copy.deepcopy(cfg)
        c.politica.margem_sleep = m
        partes = [executar_politica(g.reset_index(drop=True), POLITICAS["Green5G IA"], c, freq_min)
                  for _, g in pred.groupby("celula_id", sort=True)]
        r = contabilizar(pd.concat(partes, ignore_index=True), c.energia, c.politica, freq_min)
        s = resumir(r, c.energia, freq_min, ref)
        linhas.append({"margem_sleep": m, "economia_pct": s["economia_pct"],
                       "violacoes_qos_pct_tempo": s["violacoes_qos_pct_tempo"],
                       "usuarios_hora_afetados": s["usuarios_hora_afetados"],
                       "religamentos_por_celula_dia": s["religamentos_por_celula_dia"]})
    return linhas


def relatorio_markdown(res: dict, cfg: Config, qualidade: dict, freq_min: int) -> str:
    pg = res["previsao_geral"]
    linhas = [
        "# Green5G: relatório de avaliação (backtest walk-forward)",
        "",
        f"Dados: {qualidade['_geral']['celulas']} células, medição a cada {freq_min} min. "
        f"Teste: últimas {cfg.geral.semanas_teste} semanas, com o modelo retreinado a cada semana só com o passado.",
        "",
        "## Previsão do pico da próxima hora",
        "",
        "| Semana | Início | MAE P50 (usuários) | Cobertura P90 | Ganho vs. repetir semana passada |",
        "|---|---|---|---|---|",
    ]
    for f in res["previsao_por_semana"]:
        linhas.append(f"| {f['semana']} | {f['inicio']} | {f['mae_p50']:.1f} | {f['cobertura_p90_pct']:.1f}% "
                      f"| {f['melhoria_vs_ingenuo_pct']:.0f}% |")
    linhas += [
        f"| **Geral** | | **{pg['mae_p50']:.1f}** | **{pg['cobertura_p90_pct']:.1f}%** "
        f"| **{pg['melhoria_vs_ingenuo_pct']:.0f}%** |",
        "",
        "Cobertura P90 é a fração de vezes em que o pico real ficou abaixo do limite P90 previsto (meta: ~90%).",
        "",
        "## Comparação das políticas (todas com o mesmo painel solar)",
        "",
        "| Política | Energia da rede (kWh) | Economia | Tempo em sleep | Violações de QoS (% do tempo) "
        "| Usuários·h afetados | Religamentos/célula/dia |",
        "|---|---|---|---|---|---|---|",
    ]
    for nome, m in res["politicas"].items():
        eco = f"{m['economia_pct']:.1f}%" if "economia_pct" in m else "referência"
        linhas.append(f"| {nome} | {m['energia_rede_kwh']:.1f} | {eco} | {m['tempo_sleep_pct']:.1f}% "
                      f"| {m['violacoes_qos_pct_tempo']:.2f}% | {m['usuarios_hora_afetados']:.1f} "
                      f"| {m['religamentos_por_celula_dia']:.2f} |")
    linhas += [
        "",
        "## Sensibilidade: economia x risco (Green5G IA)",
        "",
        f"Margem de sleep = fração da capacidade da macro ({cfg.politica.capacidade_absorcao_macro:.0f} usuários) "
        f"que o pico P90 previsto precisa respeitar para a célula dormir. Configuração atual: "
        f"{cfg.politica.margem_sleep:.2f}.",
        "",
        "| Margem de sleep | Economia | Violações de QoS (% do tempo) | Usuários·h afetados | Religamentos/célula/dia |",
        "|---|---|---|---|---|",
    ]
    for s in res["sensibilidade"]:
        linhas.append(f"| {s['margem_sleep']:.2f} | {s['economia_pct']:.1f}% | {s['violacoes_qos_pct_tempo']:.2f}% "
                      f"| {s['usuarios_hora_afetados']:.1f} | {s['religamentos_por_celula_dia']:.2f} |")
    g = res["politicas"]["Green5G IA"]
    ce = cfg.energia
    linhas += [
        "",
        "## Projeção por célula (Green5G IA)",
        "",
        f"- Economia: **{g['economia_kwh_celula_ano']:.0f} kWh/ano por célula** "
        f"({g['economia_pct']:.1f}% da energia da rede; {g['economia_pct_sem_solar']:.1f}% sem considerar o solar)",
        f"- Custo evitado: R$ {g['economia_r_celula_ano']:.0f}/ano por célula (tarifa usada: R$ {ce.tarifa_r_kwh:.2f}/kWh)",
        f"- CO2 evitado: {g['co2_evitado_kg_celula_ano']:.1f} kg/ano por célula (fator: {ce.fator_co2_kg_kwh} kg/kWh)",
        f"- A conta já desconta o consumo extra da macrocélula ao absorver os usuários "
        f"({g['energia_macro_extra_kwh']:.1f} kWh no período) e o custo de cada religamento.",
        "",
        "## Qualidade dos dados",
        "",
        "| Célula | Pontos | Faltando | Preenchidos | Lacunas longas |",
        "|---|---|---|---|---|",
    ]
    for cel, q in qualidade.items():
        if cel.startswith("_"):
            continue
        linhas.append(f"| {cel} | {q['pontos']} | {q['faltando_pct']}% | {q['preenchidos']} | {q['lacunas_longas_restantes']} |")
    linhas += [
        "",
        "## Atenção",
        "",
        "Os parâmetros marcados como SUPOSIÇÃO/EXEMPLO em `green5g/config.py` (potência ociosa, capacidade de "
        "absorção da macro, custo de religamento, tarifa) mudam diretamente esses números. "
        "Substitua-os pelos valores reais antes de apresentar resultados.",
    ]
    return "\n".join(linhas) + "\n"
