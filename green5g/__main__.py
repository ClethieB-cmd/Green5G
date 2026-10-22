"""Linha de comando do Green5G.

    python -m green5g demo                                  # tudo de uma vez, com dados simulados
    python -m green5g simular  --saida dados/simulado.csv
    python -m green5g avaliar  --dados dados/simulado.csv --saida resultados/
    python -m green5g treinar  --dados dados/simulado.csv --modelo modelos/green5g.joblib
    python -m green5g decidir  --dados dados/recente.csv  --modelo modelos/green5g.joblib --estado estado.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from .config import Config


def _json_seguro(obj):
    if isinstance(obj, dict):
        return {k: _json_seguro(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_json_seguro(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _carregar(args):
    from .dados import carregar_csv
    df, freq, qualidade = carregar_csv(args.dados, args.col_tempo, args.col_celula, args.col_demanda, args.col_solar)
    print(f"Dados: {qualidade['_geral']['celulas']} células, medição a cada {freq} min, {len(df)} linhas.")
    return df, freq, qualidade


def cmd_simular(args, cfg):
    from .dados import simular_rede
    df = simular_rede(dias=args.dias, freq_min=args.freq, n_celulas=args.celulas,
                      feriados=cfg.geral.feriados, semente=args.semente)
    os.makedirs(os.path.dirname(args.saida) or ".", exist_ok=True)
    df.to_csv(args.saida, index=False)
    print(f"Simulação salva em {args.saida}: {df['celula_id'].nunique()} células, {args.dias} dias.")


def cmd_avaliar(args, cfg):
    import matplotlib
    matplotlib.use("Agg")  # linha de comando: só salva o PNG, sem abrir janelas
    from .avaliacao import backtest, relatorio_markdown
    from .dashboard import criar_dashboard
    df, freq, qualidade = _carregar(args)
    print("Backtest walk-forward (o modelo é retreinado a cada semana, só com o passado):")
    t0 = time.time()
    res = backtest(df, freq, cfg)
    os.makedirs(args.saida, exist_ok=True)
    with open(os.path.join(args.saida, "relatorio.md"), "w", encoding="utf-8") as f:
        f.write(relatorio_markdown(res, cfg, qualidade, freq))
    with open(os.path.join(args.saida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(_json_seguro({"config": cfg.para_dict(), "qualidade_dados": qualidade, **res}), f,
                  ensure_ascii=False, indent=2)
    res["_resultados"]["Green5G IA"].to_csv(os.path.join(args.saida, "decisoes_backtest.csv"), index=False)
    criar_dashboard(res, cfg, os.path.join(args.saida, "dashboard.png"))
    from .painel import gerar_painel
    perfis = (df.drop_duplicates("celula_id").set_index("celula_id")["perfil"].to_dict()
              if "perfil" in df.columns else None)
    # A coluna 'perfil' só existe nos dados do simulador: é assim que o painel sabe que os dados são simulados.
    gerar_painel(res, cfg, freq, os.path.join(args.saida, "painel.html"), perfis=perfis,
                 simulado="perfil" in df.columns)

    print(f"\n{'Política':<26}{'Economia':>10}{'Sleep':>9}{'Violações QoS':>16}{'Usuários·h afetados':>22}")
    for nome, m in res["politicas"].items():
        eco = f"{m['economia_pct']:.1f}%" if "economia_pct" in m else "ref."
        print(f"{nome:<26}{eco:>10}{m['tempo_sleep_pct']:>8.1f}%{m['violacoes_qos_pct_tempo']:>15.2f}%"
              f"{m['usuarios_hora_afetados']:>22.1f}")
    pg = res["previsao_geral"]
    print(f"\nPrevisão: erro médio {pg['mae_p50']:.1f} usuários, cobertura P90 {pg['cobertura_p90_pct']:.1f}%, "
          f"{pg['melhoria_vs_ingenuo_pct']:.0f}% melhor que repetir a semana passada.")
    print(f"Relatório, dashboard e decisões em {args.saida}/ ({time.time() - t0:.0f} s)")
    print(f"Painel web: abra {os.path.join(args.saida, 'painel.html')} no navegador")


def cmd_treinar(args, cfg):
    from .features import construir_features
    from .modelo import PrevisorDemanda
    df, freq, _ = _carregar(args)
    feats = construir_features(df, freq, cfg.politica.horizonte_min, cfg.geral.feriados)
    modelo = PrevisorDemanda(cfg.modelo, freq, cfg.politica.horizonte_min).treinar(feats)
    os.makedirs(os.path.dirname(args.modelo) or ".", exist_ok=True)
    modelo.salvar(args.modelo)
    with open(args.modelo.rsplit(".", 1)[0] + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(modelo.meta, f, ensure_ascii=False, indent=2)
    print(f"Modelo salvo em {args.modelo}")
    print(f"  treino: {modelo.meta['periodo_treino'][0]} a {modelo.meta['periodo_treino'][1]}")
    print(f"  ajuste conformal do P90: {modelo.ajuste_conformal:+.2f} usuários "
          f"(cobertura antes do ajuste: {100 * modelo.meta['cobertura_p_seg_antes_calibracao']:.1f}%)")


def cmd_decidir(args, cfg):
    from .modelo import PrevisorDemanda
    from .operacao import decidir_agora
    df, freq, _ = _carregar(args)
    modelo = PrevisorDemanda.carregar(args.modelo)
    decisoes = decidir_agora(df, freq, modelo, cfg, args.estado, args.log)
    if args.saida:
        with open(args.saida, "w", encoding="utf-8") as f:
            json.dump(decisoes, f, ensure_ascii=False, indent=2)
    for d in decisoes:
        seta = " (MUDOU)" if d["mudou"] else ""
        print(f"{d['celula_id']}: {d['modo']}{seta} a partir de {d['valido_a_partir_de']} | {d['motivo']}")


def cmd_demo(args, cfg):
    from .dados import simular_rede
    os.makedirs("dados", exist_ok=True)
    os.makedirs("resultados", exist_ok=True)
    print("== 1/4 Simulando 10 semanas de rede (6 células, medição a cada 15 min)")
    cmd_simular(argparse.Namespace(dias=70, freq=15, celulas=6, semente=42, saida="dados/simulado.csv"), cfg)
    base = dict(dados="dados/simulado.csv", col_tempo="timestamp", col_celula="celula_id",
                col_demanda="demanda", col_solar="solar_w")
    print("\n== 2/4 Avaliando")
    cmd_avaliar(argparse.Namespace(**base, saida="resultados"), cfg)
    print("\n== 3/4 Treinando o modelo final com todo o histórico")
    cmd_treinar(argparse.Namespace(**base, modelo="modelos/green5g.joblib"), cfg)
    print("\n== 4/4 Decidindo o próximo intervalo com dados novos (últimos 9 dias de uma nova simulação)")
    novo = simular_rede(dias=84, freq_min=15, n_celulas=6, feriados=cfg.geral.feriados, semente=42)
    novo = novo[novo["timestamp"] >= novo["timestamp"].max() - np.timedelta64(9, "D")]
    novo.to_csv("dados/recente.csv", index=False)
    for arq in ("estado.json", "resultados/decisoes_log.csv"):
        if os.path.exists(arq):
            os.remove(arq)
    cmd_decidir(argparse.Namespace(**{**base, "dados": "dados/recente.csv"}, modelo="modelos/green5g.joblib",
                                   estado="estado.json", log="resultados/decisoes_log.csv",
                                   saida="resultados/decisoes.json"), cfg)


def main(argv=None):
    p = argparse.ArgumentParser(prog="green5g", description="Green5G: economia de energia em células 5G com IA")
    p.add_argument("--config", help="JSON com parâmetros a sobrescrever (ver config_exemplo.json)")
    sub = p.add_subparsers(dest="comando", required=True)

    def com_dados(sp):
        sp.add_argument("--dados", required=True, help="CSV com timestamp, celula_id, demanda[, solar_w]")
        sp.add_argument("--col-tempo", default="timestamp")
        sp.add_argument("--col-celula", default="celula_id")
        sp.add_argument("--col-demanda", default="demanda")
        sp.add_argument("--col-solar", default="solar_w")

    s = sub.add_parser("simular", help="gera dados simulados realistas")
    s.add_argument("--dias", type=int, default=70)
    s.add_argument("--freq", type=int, default=15, help="minutos entre medições")
    s.add_argument("--celulas", type=int, default=6)
    s.add_argument("--semente", type=int, default=42)
    s.add_argument("--saida", default="dados/simulado.csv")

    s = sub.add_parser("avaliar", help="backtest walk-forward + relatório + dashboard")
    com_dados(s)
    s.add_argument("--saida", default="resultados")

    s = sub.add_parser("treinar", help="treina o modelo final com todo o histórico")
    com_dados(s)
    s.add_argument("--modelo", default="modelos/green5g.joblib")

    s = sub.add_parser("decidir", help="decide o próximo intervalo de cada célula")
    com_dados(s)
    s.add_argument("--modelo", default="modelos/green5g.joblib")
    s.add_argument("--estado", default="estado.json", help="arquivo com o modo atual de cada célula")
    s.add_argument("--log", default="decisoes_log.csv", help="log de auditoria (acrescenta)")
    s.add_argument("--saida", help="salva as decisões em JSON")

    sub.add_parser("demo", help="simula, avalia, treina e decide (tudo de uma vez)")

    args = p.parse_args(argv)
    cfg = Config.carregar(args.config)
    comandos = {"simular": cmd_simular, "avaliar": cmd_avaliar, "treinar": cmd_treinar,
                "decidir": cmd_decidir, "demo": cmd_demo}
    try:
        comandos[args.comando](args, cfg)
    except (ValueError, FileNotFoundError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
