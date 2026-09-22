# CLAUDE.md — Sentiment-Driven Portfolio

## Qué es este proyecto
Proyecto personal de portfolio (GitHub/CV). Un pipeline que lee noticias financieras,
las convierte en una señal de sentimiento con FinBERT y usa esa señal para asignar
pesos a una cartera de varios activos. El objetivo es aprender NLP financiero y
construcción de carteras, no batir al mercado.

Referencias (inspiración, no plantilla que haya que respetar):
- Paper: HARLF, Hierarchical RL and LLM Sentiment for Portfolio Optimization — https://arxiv.org/abs/2507.18560
- Repo: https://github.com/franjgs/llm-rl-finance-trader (FinBERT + PPO sobre UN solo valor, AAPL; la cartera multiactivo figura como trabajo futuro)
- FinBERT: https://huggingface.co/ProsusAI/finbert

## Cómo trabajo
- Hablamos en español. Código, nombres de variables y commits en inglés.
- Modo mentor: antes de escribir código, explícame el razonamiento y las alternativas en pocas líneas. Quiero entender por qué, no solo copiar.
- Respuestas concisas, organizadas por bloques funcionales.
- Avísame de forma proactiva de errores, supuestos dudosos o sesgos (sobre todo look-ahead). Si te equivocas, dilo claramente.
- No me hagas diez preguntas antes de empezar: propón un valor por defecto razonable, dilo y sigue.
- Cambios quirúrgicos: nada de funcionalidades que no haya pedido ni abstracciones para un solo uso.

## Entorno
- Trabajo a veces en Windows (PowerShell, terminal de VS Code) y a veces en Linux. Comprueba el sistema antes de darme comandos y usa la sintaxis de esa shell.
- Comandos de uno en uno, con saltos de línea. Nada de one-liners encadenados ni bucles que se rompan al pegarlos.
- Código multiplataforma: `pathlib` para rutas, sin rutas absolutas.
- Python 3.11 con `venv` y `requirements.txt` (no conda salvo que haya motivo).
- Claves de API en `.env` (nunca en el código ni en git). `.env.example` sí va al repo.

## Reglas del dominio (no negociables)
1. **Sin look-ahead.** Una noticia publicada en el día t solo puede afectar a la decisión que se ejecuta en t+1 (o después del cierre, según su timestamp). Cualquier join noticias–precios debe dejar esto explícito y con test.
2. **Split temporal.** Train / validación / test por fechas, nunca aleatorio. El test no se toca hasta el final.
3. **Siempre contra baselines.** Toda estrategia se compara con equal-weight y buy-and-hold del índice, con costes de transacción incluidos.
4. **Métricas mínimas:** rentabilidad anualizada, volatilidad, Sharpe, máximo drawdown, turnover.
5. **Honestidad estadística.** Con uno o dos años de datos diarios, una diferencia de Sharpe no demuestra nada por sí sola. Dilo cuando toque.
6. Cachear en disco la salida de FinBERT (es lo caro) para no recalcularla en cada ejecución.

## Estado del proyecto
(Actualizar al cerrar cada sesión: qué fase está hecha, qué decisiones se tomaron y por qué, qué queda pendiente.)
- Fase 0 (planificación): HECHA el 2026-09-22. El plan completo está en `PLAN.md`.
- Decisiones clave (detalle y motivos en PLAN.md §3 y §6):
  - Noticias: Alpaca News API (Benzinga), de 2016-01 a 2026-08, con `created_at` UTC. FNSPID se descartó por timestamps poco fiables y porque termina en 2023.
  - Universo ex-ante (anti-supervivencia), verificado con SEC EDGAR: AAPL, GOOGL, AMZN, PG, JNJ, WFC, XOM, BA y DUK. WFC sustituye a JPM porque era mayor a 2015-12.
  - Split: train 2016–2021, validación 2022–2023, test 2024-01 → 2026-08. El test se abre una sola vez, en la fase 7.
  - Timing: noticias con `created_at` anterior al cierre NYSE de t entran en t. La decisión se toma el último día de la semana y se ejecuta al cierre de la sesión siguiente.
  - Señal: P_pos − P_neg del titular (vía `id2label`), media de N sesiones y z-score transversal. Regla de tilt con límites [0,5/n, 2/n].
  - Costes de 10 pb por dólar negociado. Sharpe sobre `^IRX`. Controles: placebo y momentum.
  - PPO (fase 6) fuera del MVP; solo entra si pasa la puerta (IC con t > 2 y el tilt bate a EW en validación).
- Fase 1 (esqueleto, precios, motor y baselines): HECHA el 2026-09-22.
  - Convención del motor: la cartera se forma al cierre de la primera ejecución sin coste (igual para todas las estrategias) y los resultados empiezan en la sesión siguiente. Coste = c·Σ|Δw|; el turnover solo cuenta rebalanceos.
  - Train (Sharpe): EW 0,97 y SPY 0,98. Validación: EW 0,29 y SPY 0,00. El coste pesa poco en EW: el Sharpe pasa de 0,81 a 0,79 entre 0 y 25 pb.
  - Los precios de 2024–2026 están en disco, pero ningún script los lee todavía (cortan en el fin de validación).
- Fase actual: 2 (descarga de noticias de Alpaca y EDA). Necesita las claves de Alpaca en `.env`.
