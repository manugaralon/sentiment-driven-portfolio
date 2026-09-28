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
- Fase 2 (noticias), en curso. Decisiones del 2026-09-22:
  - Verificado: Alpaca filtra `start`/`end` por `updated_at`, no por `created_at`. Se descarga hasta el mes actual y se filtra por `created_at` al cargar.
  - Verificado: una sola petición con todos los tickers devuelve exactamente la unión de las peticiones individuales.
  - Cobertura muy desigual: AAPL, AMZN y GOOGL tienen más de 100 artículos al mes; PG entre 1 y 14 y DUK entre 0 y 2. **Los tickers sin noticias en la ventana reciben z = 0 (peso EW)**, y el z-score se calcula solo entre los que tienen noticias; así se evita infraponderar por falta de cobertura. El universo se mantiene.
  - Descarga completa: 88.451 artículos (2015-10 → 2026-09, 17 MB), en 26 min. En train + validación hay 59.640 únicos.
  - EDA (`scripts/news_eda.py`, solo train + validación). Decisiones:
    - **Relevancia:** el titular nombra a la empresa **y** el artículo lleva ≤ 5 símbolos (`news.max_symbols`). Conserva el 91 % de los artículos con nombre y descarta las listas ("Top 10 stocks…").
    - **Timestamp de disponibilidad = `updated_at`, no `created_at`:** el titular que tenemos es la versión editada. Descartar los editados sería en sí una selección con look-ahead. Solo retrasa unas horas un ~2 % de artículos (un 4 % en 2023). Es más estricto que el plan original.
    - Cobertura (% de sesiones con al menos una noticia relevante): AAPL 95, AMZN 92, GOOGL 79, BA 69, JNJ 46, XOM 44, WFC 38, PG 20 y DUK 17. Crece con el tiempo y con los eventos (737 MAX, vacuna, 2022 energía).
    - Horas ET: 34 % preapertura, 52 % en sesión, 10,5 % tras el cierre y 4 % en fin de semana. La zona horaria es correcta (los "pre-market" tienen mediana a las 8 h).
    - Boilerplate: ratings de analistas 7,5 %, "options activity" 4 % y earnings 8 %. No se excluye ninguna categoría; las listas ya caen por el filtro de símbolos.
    - Se mantienen el inicio en 2016-01 y el grid N ∈ {5, 10, 20}.
- Fase 3 (FinBERT, caché y alineación): HECHA el 2026-09-28.
  - 45.896 titulares únicos relevantes (2015-10 → 2026-08) puntuados y cacheados; al relanzar, el 100 % sale de caché. Unos 14 titulares/s en CPU con lotes ordenados por longitud.
  - Train: media +0,05; un 31 % en [−0,1, 0,1]. Sesgos por ticker: DUK +0,16, JNJ +0,16, WFC −0,11. Los ratings de analistas puntúan +0,28 de media.
  - Ruido visto en la revisión manual: titulares de movimiento de precio ("Shares Fall…") que van *detrás* del precio, y algún error ("Amazon Breaches $500" sale −0,91).
  - Hook de pre-commit activado (`.claude/quality-gate-precommit`, ignorado en git). Su lint se salta porque ruff no está en el PATH global.
- Fase 4 (señal y análisis del IC en train): HECHA el 2026-09-28.
  - `signals.py`, `stats.py`, `scripts/build_signal.py` y `tests/test_signal.py`. El test de truncado pasa y detecta un look-ahead metido a propósito. `data/processed/signals.parquet` guarda 6 señales (raw/surprise × N) para todo el periodo.
  - Variante **sorpresa**: resta la media propia del ticker con los artículos *anteriores* a la ventana; exige ≥ 20 artículos. Con menos de 3 tickers con noticias en una sesión, z = 0 para todos. IC95 con Newey-West, lag = h + N.
  - Bug corregido: `unstack()` dejaba NaN en lugar de 0 en el panel y apagaba la variante sorpresa.
  - Resultado en train: IC a futuro ≈ 0 y **negativo en las 18 combinaciones**. El mejor es surprise N=5, h=20, con −0,05 y t = −2,7. Con la rentabilidad **pasada** es fuertemente positivo (+0,09 a +0,17, t de 5 a 8): las noticias van detrás del precio. La rentabilidad pasada por sí sola no predice (|t| < 1,7), así que el signo negativo no es reversión de precio.
  - Decisión para la fase 5 (2026-09-28): **no se invierte el signo** (sería data snooping). La fase 5 sigue el plan, con tilt positivo y el grid completo; lo esperado es que no bata a EW. El tilt contrario (λ < 0) va solo como análisis exploratorio etiquetado, confirmado en validación, y no cuenta para la puerta de la fase 6. Variante: **sorpresa**, por diseño (evita el tilt fijo hacia DUK y JNJ), no por su IC.
- Fase actual: 5 (estrategia de tilt en train y validación).
