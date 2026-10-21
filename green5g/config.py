"""Parâmetros do Green5G.

Tudo que é número de negócio ou de equipamento fica aqui, num só lugar.
Valores marcados com SUPOSIÇÃO ou EXEMPLO precisam ser trocados pelos
dados reais (ficha técnica do equipamento, conta de energia, fator de
emissão oficial, capacidade real da macrocélula) antes de um piloto.

Qualquer campo pode ser sobrescrito por um arquivo JSON:
    python -m green5g avaliar --dados x.csv --config minha_config.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass
class ConfigEnergia:
    # Modelo de potência dependente da carga (mesma ideia do modelo EARTH):
    #   P = P_ociosa + (P_max - P_ociosa) * carga,   carga em [0, 1]
    potencia_max_w: float = 850.0              # célula a 100% de carga (valor do projeto original)
    potencia_ociosa_w: float = 550.0           # SUPOSIÇÃO: ligada sem usuários (usar ficha técnica)
    potencia_sleep_w: float = 75.0             # modo sleep (valor do projeto original)
    capacidade_celula_usuarios: float = 250.0  # carga em que a célula chega à potência máxima
    energia_religamento_wh: float = 30.0       # SUPOSIÇÃO: custo de energia de cada religamento
    custo_macro_w_por_usuario: float = 3.0     # SUPOSIÇÃO: consumo extra da macro por usuário absorvido
    eficiencia_solar: float = 0.82             # inversor + perdas
    tarifa_r_kwh: float = 0.85                 # EXEMPLO: substitua pela tarifa da sua conta (com impostos)
    fator_co2_kg_kwh: float = 0.0247           # MCTI, média mensal jan-abr/2025 (24,7 kg/MWh). Atualizar.


@dataclass
class ConfigPolitica:
    horizonte_min: int = 60                    # a decisão olha o pico da próxima 1 h
    capacidade_absorcao_macro: float = 45.0    # SUPOSIÇÃO: usuários que a macro absorve sem degradar QoS
    margem_sleep: float = 0.85                 # dorme se P90 do pico < 85% da capacidade de absorção
    margem_acordar: float = 1.00               # acorda se P90 do pico > 100% da capacidade
    permanencia_min_ativa_min: int = 60        # anti-oscilação: fica ativa ao menos 60 min antes de dormir
    cobertura_minima: float = 0.75             # se o P90 acertar menos que isso -> modo seguro
    janela_monitor_h: int = 72                 # janela do monitor de calibração
    amostras_min_monitor: int = 24             # amostras mínimas antes de o monitor atuar
    horas_protegidas: list[int] = field(default_factory=list)  # regra da operadora: nunca dormir nessas horas


@dataclass
class ConfigModelo:
    quantil_seguranca: float = 0.90
    dias_calibracao: int = 7                   # última semana do treino usada para a calibração conformal
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    semente: int = 42


@dataclass
class ConfigGeral:
    semanas_teste: int = 3                     # semanas finais usadas no backtest walk-forward
    feriados: list[str] = field(default_factory=lambda: [
        "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-15", "2026-11-20", "2026-12-25",
    ])


@dataclass
class Config:
    energia: ConfigEnergia = field(default_factory=ConfigEnergia)
    politica: ConfigPolitica = field(default_factory=ConfigPolitica)
    modelo: ConfigModelo = field(default_factory=ConfigModelo)
    geral: ConfigGeral = field(default_factory=ConfigGeral)

    def para_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def carregar(cls, caminho: str | None = None) -> "Config":
        cfg = cls()
        if not caminho:
            return cfg
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)
        for secao in fields(cls):
            if secao.name not in dados:
                continue
            alvo = getattr(cfg, secao.name)
            validos = {f.name for f in fields(alvo)}
            for chave, valor in dados[secao.name].items():
                if chave not in validos:
                    raise ValueError(f"Parâmetro desconhecido em '{secao.name}': {chave}")
                setattr(alvo, chave, valor)
        return cfg
