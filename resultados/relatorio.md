# Green5G: relatório de avaliação (backtest walk-forward)

Dados: 6 células, medição a cada 15 min. Teste: últimas 3 semanas, com o modelo retreinado a cada semana só com o passado.

## Previsão do pico da próxima hora

| Semana | Início | MAE P50 (usuários) | Cobertura P90 | Ganho vs. repetir semana passada |
|---|---|---|---|---|
| 1 | 2026-09-21 | 5.6 | 92.0% | 39% |
| 2 | 2026-09-28 | 6.3 | 86.3% | 41% |
| 3 | 2026-10-05 | 5.9 | 91.7% | 35% |
| **Geral** | | **5.9** | **90.0%** | **39%** |

Cobertura P90 é a fração de vezes em que o pico real ficou abaixo do limite P90 previsto (meta: ~90%).

## Comparação das políticas (todas com o mesmo painel solar)

| Política | Energia da rede (kWh) | Economia | Tempo em sleep | Violações de QoS (% do tempo) | Usuários·h afetados | Religamentos/célula/dia |
|---|---|---|---|---|---|---|
| Sempre ativa | 1277.6 | referência | 0.0% | 0.00% | 0.0 | 0.00 |
| Regras v1 (original) | 957.8 | 25.0% | 24.6% | 0.27% | 31.0 | 1.08 |
| Green5G IA | 975.8 | 23.6% | 23.4% | 0.01% | 1.1 | 1.06 |
| Oráculo (limite teórico) | 892.0 | 30.2% | 36.8% | 0.00% | 0.0 | 1.60 |

## Sensibilidade: economia x risco (Green5G IA)

Margem de sleep = fração da capacidade da macro (45 usuários) que o pico P90 previsto precisa respeitar para a célula dormir. Configuração atual: 0.85.

| Margem de sleep | Economia | Violações de QoS (% do tempo) | Usuários·h afetados | Religamentos/célula/dia |
|---|---|---|---|---|
| 0.70 | 21.6% | 0.00% | 0.0 | 0.98 |
| 0.80 | 22.9% | 0.01% | 1.1 | 1.00 |
| 0.85 | 23.6% | 0.01% | 1.1 | 1.06 |
| 0.90 | 24.3% | 0.01% | 1.1 | 1.19 |
| 1.00 | 25.2% | 0.06% | 4.5 | 1.53 |

## Projeção por célula (Green5G IA)

- Economia: **874 kWh/ano por célula** (23.6% da energia da rede; 16.4% sem considerar o solar)
- Custo evitado: R$ 743/ano por célula (tarifa usada: R$ 0.85/kWh)
- CO2 evitado: 21.6 kg/ano por célula (fator: 0.0247 kg/kWh)
- A conta já desconta o consumo extra da macrocélula ao absorver os usuários (37.4 kWh no período) e o custo de cada religamento.

## Qualidade dos dados

| Célula | Pontos | Faltando | Preenchidos | Lacunas longas |
|---|---|---|---|---|
| CEL-01 | 6720 | 0.73% | 37 | 12 |
| CEL-02 | 6720 | 0.64% | 31 | 12 |
| CEL-03 | 6720 | 0.7% | 35 | 12 |
| CEL-04 | 6720 | 0.73% | 37 | 12 |
| CEL-05 | 6720 | 0.74% | 38 | 12 |
| CEL-06 | 6720 | 0.67% | 33 | 12 |

## Atenção

Os parâmetros marcados como SUPOSIÇÃO/EXEMPLO em `green5g/config.py` (potência ociosa, capacidade de absorção da macro, custo de religamento, tarifa) mudam diretamente esses números. Substitua-os pelos valores reais antes de apresentar resultados.
