"""Gráficos do Green5G.

Cada gráfico é uma função que desenha num eixo (ax), para ser usada tanto no
dashboard PNG da linha de comando quanto em notebooks (Colab/Jupyter).
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .politica import SLEEP

COR = {
    "fundo": "#fcfcfb", "texto": "#0b0b0b", "texto2": "#52514e", "grade": "#e4e3df",
    "azul": "#2a78d6", "azul_claro": "#b7d3f6", "laranja": "#eb6834", "aqua": "#1baf7a", "cinza": "#a9a8a3",
}


def estilo(ax, titulo):
    ax.set_facecolor(COR["fundo"])
    ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold", color=COR["texto"], pad=10)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(COR["grade"])
    ax.tick_params(colors=COR["texto2"], labelsize=9)
    ax.grid(True, color=COR["grade"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def grafico_celula(ax, res: dict, cfg, celula: str | None = None, dias: int = 3, inicio=None):
    """Demanda real, faixa de previsão P50-P90 e períodos em sleep de uma célula."""
    green = res["_resultados"]["Green5G IA"]
    celula = celula or sorted(green["celula_id"].unique())[0]
    cap = cfg.politica.capacidade_absorcao_macro
    g = green[green["celula_id"] == celula].sort_values("timestamp")
    ini = pd.Timestamp(inicio) if inicio is not None else g["timestamp"].min()
    g = g[(g["timestamp"] >= ini) & (g["timestamp"] < ini + pd.Timedelta(days=dias))]
    if g.empty:
        raise ValueError(f"Sem dados de teste para {celula} a partir de {ini}.")
    t = g["timestamp"]
    topo = np.nanmax([g["demanda"].max(), g["p90"].max(), cap]) * 1.1
    ax.fill_between(t, 0, topo, where=(g["modo"] == SLEEP).to_numpy(), step="post", color=COR["aqua"],
                    alpha=0.18, linewidth=0, label="Célula em sleep")
    ax.fill_between(t, g["p50"], g["p90"], color=COR["azul_claro"], alpha=0.7, linewidth=0,
                    label="Previsão do pico da próxima hora (P50 a P90)")
    ax.plot(t, g["demanda"], color=COR["azul"], linewidth=2, label="Demanda real")
    ax.axhline(cap, color=COR["texto2"], linestyle="--", linewidth=1.2,
               label=f"Capacidade de absorção da macro ({cap:.0f})")
    ax.set_ylim(0, topo)
    ax.set_ylabel("Usuários ativos", color=COR["texto2"])
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    estilo(ax, f"{celula}: {dias} dias de operação")


def grafico_consumo(ax, res: dict):
    """Energia da rede por política, com a economia em relação a 'sempre ativa'."""
    nomes = list(res["politicas"])
    valores = [res["politicas"][n]["energia_rede_kwh"] for n in nomes]
    y = np.arange(len(nomes))[::-1]
    ax.barh(y, valores, height=0.55,
            color=[COR["azul"] if n == "Green5G IA" else COR["cinza"] for n in nomes])
    for yi, n, v in zip(y, nomes, valores):
        m = res["politicas"][n]
        extra = f"  -{m['economia_pct']:.1f}%" if "economia_pct" in m else "  referência"
        ax.text(v, yi, f"  {v:.0f} kWh{extra}", va="center", fontsize=9.5, color=COR["texto"])
    ax.set_yticks(y)
    ax.set_yticklabels(nomes, color=COR["texto"], fontsize=10)
    ax.set_xlim(0, max(valores) * 1.35)
    ax.set_xlabel("kWh da rede elétrica no período de teste (todas com o mesmo solar)", color=COR["texto2"])
    estilo(ax, "Consumo por política")
    ax.grid(axis="y", visible=False)


def grafico_risco(ax, res: dict, cfg):
    """Curva economia x usuários afetados ao variar a margem de sleep."""
    sens = pd.DataFrame(res["sensibilidade"])
    ax.plot(sens["usuarios_hora_afetados"], sens["economia_pct"], color=COR["azul"], linewidth=2,
            marker="o", markersize=8, label="Green5G IA (variando a margem de sleep)")
    for _, s in sens.iterrows():
        atual = " (atual)" if abs(s["margem_sleep"] - cfg.politica.margem_sleep) < 1e-9 else ""
        ax.annotate(f"margem {s['margem_sleep']:.2f}{atual}", (s["usuarios_hora_afetados"], s["economia_pct"]),
                    textcoords="offset points", xytext=(8, -12), fontsize=8.5, color=COR["texto2"])
    for nome, marcador, cor in (("Regras v1 (original)", "s", COR["laranja"]),
                                ("Oráculo (limite teórico)", "D", COR["texto2"])):
        m = res["politicas"][nome]
        ax.scatter(m["usuarios_hora_afetados"], m["economia_pct"], s=70, marker=marcador, color=cor,
                   zorder=3, label=nome)
    ax.set_xlabel("Usuários·hora afetados (sleep com demanda acima da capacidade da macro)", color=COR["texto2"])
    ax.set_ylabel("Economia de energia (%)", color=COR["texto2"])
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    estilo(ax, "Economia x risco de QoS")


def grafico_erro_hora(ax, res: dict):
    """Erro da previsão por hora do dia: IA vs. repetir a semana passada."""
    v = res["_pred"].dropna(subset=["alvo", "max_janela_semana"]).copy()
    v["hora"] = v["timestamp"].dt.hour
    v["erro_ia"] = (v["alvo"] - v["p50"]).abs()
    v["erro_ing"] = (v["alvo"] - v["max_janela_semana"]).abs()
    por_hora = v.groupby("hora")[["erro_ia", "erro_ing"]].mean()
    ax.plot(por_hora.index, por_hora["erro_ing"], color=COR["laranja"], linewidth=2,
            label="Repetir a semana passada")
    ax.plot(por_hora.index, por_hora["erro_ia"], color=COR["azul"], linewidth=2, label="Green5G IA (P50)")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hora do dia", color=COR["texto2"])
    ax.set_ylabel("Erro médio absoluto (usuários)", color=COR["texto2"])
    ax.set_ylim(0, None)
    pg = res["previsao_geral"]
    ax.legend(loc="upper left", fontsize=8.5, frameon=False,
              title=f"Cobertura do P90: {pg['cobertura_p90_pct']:.1f}% (meta 90%)", title_fontsize=8.5)
    estilo(ax, "Erro da previsão do pico da próxima hora")


def grafico_mapa_sleep(ax, res: dict, politica: str = "Green5G IA"):
    """Mapa de calor: % do tempo em sleep por hora do dia e dia da semana."""
    r = res["_resultados"][politica]
    tab = (r.assign(hora=r["timestamp"].dt.hour, dia=r["timestamp"].dt.dayofweek,
                    dormindo=(r["modo"] == SLEEP))
             .pivot_table(index="dia", columns="hora", values="dormindo", aggfunc="mean")
             .reindex(index=range(7), columns=range(24)) * 100)
    im = ax.imshow(tab.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["seg", "ter", "qua", "qui", "sex", "sáb", "dom"], color=COR["texto"])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora do dia", color=COR["texto2"])
    estilo(ax, f"% do tempo em sleep ({politica})")
    ax.grid(False)
    barra = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    barra.set_label("% do tempo em sleep", color=COR["texto2"])
    barra.outline.set_visible(False)


def criar_dashboard(res: dict, cfg, caminho: str, celula: str | None = None, dias_exibidos: int = 3) -> None:
    fig, eixos = plt.subplots(2, 2, figsize=(16, 10), facecolor=COR["fundo"])
    fig.suptitle("Green5G: previsão de demanda e economia de energia (backtest)", x=0.02, ha="left",
                 fontsize=16, fontweight="bold", color=COR["texto"])
    grafico_celula(eixos[0, 0], res, cfg, celula, dias_exibidos)
    grafico_consumo(eixos[0, 1], res)
    grafico_risco(eixos[1, 0], res, cfg)
    grafico_erro_hora(eixos[1, 1], res)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(caminho, dpi=110, facecolor=COR["fundo"])
    plt.close(fig)
