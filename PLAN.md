# PLAN: Sentiment-Driven Portfolio

## Context
Proyecto personal de portfolio para CV: una cartera de 9 acciones grandes de EE. UU. con pesos semanales que dependen de una señal diaria de sentimiento. La señal se calcula con FinBERT sobre titulares de noticias. El objetivo es aprender NLP financiero y construcción de carteras con rigor: sin look-ahead, split temporal, baselines con costes y honestidad estadística. Batir al mercado no es el objetivo.

Plan elaborado y aprobado en la fase 0 (2026-09-22).

Hardware verificado (Linux): 4 CPU, ~10 GB de RAM (~4,5 GB libres), sin GPU y 40 GB libres en disco. FinBERT tendrá que correr en CPU, así que la caché por artículo tiene que ser incremental y reanudable. Python 3.11 está disponible en `~/.local/bin/python3.11`.

Leyenda: **[V]** comprobado hoy en la documentación o el código. **[NV]** no verificado: sale de la memoria, de terceros o es una estimación.

---

## 1. Referencias: qué aprovecho y qué no

### Repo franjgs/llm-rl-finance-trader [V: leídos README, config, `data_fetch.py`, `sentiment_analysis.py`, `src/trading_env.py`, `train_model.py`]

**Aprovecho:**
- La forma del pipeline: fetch → score → env → train.
- La caché de noticias por símbolo, deduplicada (rama `develop`).
- El walk-forward expansivo de `develop`, como patrón de evaluación.

**No aprovecho:**
- **Tiene el signo de FinBERT invertido.** Lee `pos=probs[1]`, pero `id2label` es `{0: positive, 1: negative, 2: neutral}`. Lección: mapear siempre con `model.config.id2label`.
- **Tiene look-ahead el mismo día.** Las noticias del día t, incluidas las publicadas tras el cierre, deciden la operación al cierre de t. Además, las noticias de fin de semana se descartan.
- Descarta la hora de publicación y concatena 3 titulares por día.
- Da una puntuación positiva a los días neutros.
- La recompensa es el efectivo recibido al vender (premia rotar la cartera) y no hay costes.
- No hay split: el Sharpe del README (0,66 frente a 0,25) es in-sample y de una sola semilla.
- Hay bugs de ficheros y de configuración que hacen que el agente "con sentimiento" probablemente reciba ceros.
- Opera un solo valor con acción discreta, así que no sirve para una cartera.

### Paper HARLF, arXiv 2507.18560 [V: abstract, HTML y los Colab publicados]

**Aprovecho:**
- La puntuación continua S = media(P_pos − P_neg) de la fórmula del paper, en vez de quedarse con la etiqueta top.
- La idea de darle al modelo la volatilidad junto al sentimiento.
- Su desfase bien hecho: la observación del mes k decide y la recompensa se mide en k+1.
- Reportar la mediana de varias semillas.

**No aprovecho:**
- **Su resultado estrella está contaminado [V en código].** Los meta y super-agentes se entrenan con backtests de 2003 a 2024 y se evalúan en 2018–2024, así que es in-sample.
- Los pesos se multiplican por **precios en nivel**, no por rentabilidades.
- Usa costes cero y no tiene conjunto de validación.
- Scrapear Google News: el ranking es el de hoy y las páginas pueden estar editadas, lo que es look-ahead en la recuperación [NV].
- La jerarquía de 3 niveles de RL, que es excesiva para los datos que tenemos.
- Su universo son 14 índices y materias primas con rebalanceo mensual, no acciones.

**Nota a favor de FinBERT:** su preentrenamiento (Reuters 2008–2010 y Financial PhraseBank) es anterior a nuestro periodo, así que el modelo no "sabe" el futuro. Un LLM moderno sí tendría look-ahead por sus datos de entrenamiento.

---

## 2. Datos de noticias

| Fuente | Histórico en plan gratuito | Límites | ¿Hora? | Contenido | Clave |
|---|---|---|---|---|---|
| **Alpaca** `/v1beta1/news` | Desde 2015 [V] | 200 llamadas/min, 50 artículos por llamada, paginación con `page_token` [V] | Sí: `created_at` en RFC-3339 UTC [V] | Titular, resumen, cuerpo HTML y `symbols`. Solo Benzinga [V] | Sí; basta una cuenta paper gratuita [V] |
| Finnhub `/company-news` | **1 año** [V] | 60/min [V] | Sí, UNIX [V] | Titular y resumen, multifuente | Sí. El ToS prohíbe compartir incluso resultados derivados [V] |
| Massive (antes Polygon) | **2 años** [V] | 5/min [V] | Sí, UTC [V] | Título y descripción | Sí; solo uso personal [V] |
| Alpha Vantage `NEWS_SENTIMENT` | Desde ~2022-03 [NV] | **25 al día** [V] | Sí [V] | Título, resumen y su propio sentimiento | Sí |
| GDELT DOC 2.0 | Desde 2017 [V] | 1 petición cada 5 s, 250 registros máx. [V] | Sí, en pasos de 15 min [V] | Solo título y URL. Busca por nombre, no por ticker | No |
| **FNSPID** (HF `Zihan1004/FNSPID`) | 1999–2023, según el paper [NV]. La cobertura por año no la pude sacar: el índice de HF no cargó | Descarga de CSV: 5,7 GB (`All_external`) y 23 GB (`nasdaq`) [V] | **Parcial** [V en muestra] | Título, URL, editor, a veces cuerpo y resúmenes. Benzinga y Nasdaq | No. Licencia CC BY-NC 4.0 [V] |
| Tiingo, NewsAPI, Marketaux, EODHD, StockNewsAPI | 1–3 meses, o sin plan gratuito útil [V] | – | – | – | – |

**Hallazgo sobre FNSPID [V en muestra, interpretación NV]:**
- Muchos timestamps son `00:00:00 UTC`, es decir, solo fecha.
- En los artículos de Benzinga las horas parecen desplazadas unas −4 h respecto a la hora de publicación real: "46 Stocks Moving In Friday's Mid-Day Session" aparece a las 08:45 "UTC".
- Con una hora que no es fiable, la regla antes/después del cierre no se puede aplicar. Además, el dataset se acaba en 2023.

**Recomendación: Alpaca News (Benzinga), de 2016-01 a 2026-08.** Es la única fuente gratuita con más de 3 años, con ticker y con hora exacta.
- Estimación de descarga: ~300.000 artículos en ~30 minutos [NV]. Un piloto de un mes lo medirá.
- Contras:
  - Una sola editorial, con mucho boilerplate ("Maintains Buy, raises PT", "Unusual Options Activity", listas de "stocks moving").
  - El filtro `start`/`end` puede ir por fecha de actualización [NV]. La alineación usará siempre `created_at`.
  - No hay términos de redistribución claros, así que `data/` no se sube a git.
- FNSPID queda como alternativa sin clave. Solo lo usaría si Alpaca fallara, y tratando las horas como desconocidas (retardo de un día completo).

---

## 3. Alcance del MVP (cambios sobre tus valores por defecto, con motivo)

- **Universo (cambio): elegido ex-ante, no con los grandes de hoy.** Con los grandes de hoy, el universo son ganadores ex-post, y eso premia cualquier señal correlacionada con "a esta empresa le va bien" (las noticias positivas). Es sesgo de supervivencia que favorece justo a nuestra estrategia.
  - Regla: la mayor capitalización de cada sector GICS a 2016-01, excluyendo conglomerados y empresas con spin-offs o fusiones grandes en el periodo (complican los precios ajustados).
  - Propuesta, 9 nombres: **AAPL, GOOGL, AMZN, PG, JNJ, JPM, XOM, BA, DUK o NEE**.
  - Quedan fuera BRK.B, GE y MMM por conglomerados; Materials por fusiones (DD/DOW); y Real Estate por tener poca cobertura.
  - El ranking exacto a 2016-01 lo verifico en la fase 1 [NV].
- **Periodo y split:**
  - Train: 2016-01 → 2021-12 (6 años). Aquí se hacen el análisis de la señal y el grid.
  - Validación: 2022-01 → 2023-12 (incluye el bear market de 2022).
  - Test: 2024-01 → 2026-08. Se abre **una sola vez, en la última fase, para todas las estrategias a la vez**.
- **Timing: diario, con rebalanceo semanal (lo mantengo).**
  - Un artículo entra en la información de la sesión t si `created_at` < cierre de t en hora ET. Uso el calendario NYSE real, con festivos y medias sesiones de cierre a las 13:00.
  - Lo publicado después del cierre y en fin de semana pasa a la siguiente sesión.
  - La decisión se toma al cierre de la última sesión de la semana y se **ejecuta al cierre de la sesión siguiente**. Es conservador y solo necesita el cierre ajustado.
- **Señal:**
  - Puntuación por titular: P_pos − P_neg vía `id2label`. Solo el titular.
  - Se descartan los artículos etiquetados con más de K=3 símbolos (listas de valores). K se revisa en el EDA.
  - Por ticker: media de las puntuaciones de los artículos de las últimas N sesiones (0 si no hay noticias), y después z-score transversal recortado a ±2.
- **Regla (b):**
  - w_i = (1/n)·(1 + λ·z_i), recortado a [0,5/n, 2/n] y renormalizado. Long-only y totalmente invertido.
  - Grid de 9 combinaciones: N ∈ {5, 10, 20} y λ ∈ {0,25, 0,5, 1}. Se elige en train, se confirma en validación y se reporta el grid entero, no solo el mejor.
- **Baselines y controles:**
  - Equal-weight rebalanceado cada semana con los mismos costes, y buy-and-hold de SPY.
  - Controles añadidos (baratos, reutilizan la misma función de tilt):
    - **Placebo:** la señal permutada entre tickers unas 200 veces, para obtener un p-valor empírico.
    - **Tilt por momentum:** las noticias suelen ir detrás del precio, así que hay que comprobar si el sentimiento aporta algo más allá del momentum.
- **Costes:** 10 pb por dólar negociado (c·Σ|Δw| sobre los pesos ya desplazados por el mercado). Análisis de sensibilidad con 0, 5 y 25 pb.
- **Métricas:** rentabilidad anualizada, volatilidad, Sharpe (sobre el T-bill de 3 meses, `^IRX`), máximo drawdown, turnover anual e information ratio frente a equal-weight.
- **Estadística:** con 2,7 años de test, el error estándar del Sharpe es de ≈0,7. Comparar Sharpes sueltos no demuestra nada.
  - Lo informativo es el test **pareado** sobre las rentabilidades activas (tilt − EW). Si la correlación con EW es ~0,95, el error estándar de la diferencia de Sharpe baja a ≈0,25 (aproximación Jobson-Korkie/Memmel).
  - Complementos: bootstrap por bloques de esa diferencia, y el IC sobre todo el periodo de train.
  - Expectativa honesta: con 9 nombres y ~310 semanas de train, el error estándar del IC medio es ≈0,02. Hace falta un IC ≥0,04 para que sea significativo, y lo esperable está por debajo.

### Veredicto sobre la fase c (PPO)
**Con estos datos, no.** Sobreajustaría:
- Hay ~310 decisiones semanales de train (~1.500 pasos diarios), una acción continua de 9 dimensiones y una relación señal/ruido minúscula.
- PPO acabaría memorizando una única trayectoria histórica, y la varianza entre semillas superaría a la diferencia entre estrategias.
- HARLF es el ejemplo de cómo sale "bien" solo con fuga de test.

**Propuesta:**
- La fase 6 queda como opcional y con una puerta: IC en train con t > 2 **y** que el tilt bata a EW en validación neto de costes.
- Si no la pasa (lo probable), la saltamos y lo documentamos. Un resultado negativo bien hecho vale más en el CV que un Sharpe de 3 sobreajustado.
- Si la pasa: al menos 5 semillas y mediana ± dispersión. El agente se evalúa en el mismo test único que el resto.

---

## 4. Estructura del repo
```
sentiment_driven_portfolio/
├── CLAUDE.md  PLAN.md  README.md
├── requirements.txt          # dependencias; torch CPU se instala aparte (ver fase 1)
├── pyproject.toml            # solo instalación editable + configuración de pytest y ruff
├── .env.example  .gitignore  # ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY; data/ y .env ignorados
├── config/config.yaml        # universo, fechas, split, costes, grid, revisión del modelo
├── data/                     # entera en .gitignore
│   ├── raw/prices/  raw/news/
│   ├── cache/finbert/        # puntuación por id de artículo, con la revisión del modelo en la clave
│   └── processed/            # panel diario, señales
├── src/sentiment_portfolio/
│   ├── config.py  prices.py  news.py
│   ├── alignment.py          # timestamp → sesión con información (núcleo anti look-ahead)
│   ├── finbert.py  signals.py  strategies.py
│   ├── backtest.py  metrics.py  stats.py
│   └── rl_env.py             # solo si se pasa la puerta de la fase 6
├── scripts/                  # un paso del pipeline por script
│   ├── download_prices.py  download_news.py  score_news.py
│   ├── build_signal.py  run_backtest.py  (train_ppo.py)
├── notebooks/                # EDA y análisis; la lógica vive en src/
├── tests/
└── reports/figures/
```

---

## 5. Plan por fases

### Fase 1: Esqueleto, precios, motor de backtest y baselines
- **Objetivo:** que el repo sea ejecutable, con precios cacheados y un backtest de equal-weight semanal y de SPY con costes y métricas.
- **Ficheros:** `.gitignore`, `.env.example`, `requirements.txt`, `pyproject.toml`, `config/config.yaml`, `src/sentiment_portfolio/{config,prices,backtest,metrics}.py`, `scripts/download_prices.py`, `scripts/run_backtest.py`, `tests/test_backtest.py`, `tests/test_metrics.py`, `README.md` (esqueleto). Además, `git init` y el primer commit.
- **Hecho cuando:**
  - `pytest` pasa en verde con tests de respuesta conocida:
    - Métricas exactas con una serie de rentabilidad constante.
    - Equal-weight de 2 activos igual al cálculo a mano.
    - Coste = c·turnover exacto.
    - Los pesos derivan correctamente entre rebalanceos.
  - Existe la figura `reports/figures/baselines_trainval.png` con EW frente a SPY entre 2016 y 2023.
  - Ranking del universo verificado y anotado en el config.
- **Riesgos:**
  - yfinance es inestable, así que todo se cachea en parquet.
  - El precio ajustado incluye dividendos (rentabilidad total), lo que es coherente para SPY y para las acciones.
  - En Linux, `pip install torch` baja CUDA (varios GB). Se instala con el índice CPU de PyTorch como paso documentado.

### Fase 2: Descarga de noticias y EDA
- **Objetivo:** tener las noticias de Alpaca de 2016-01 a 2026-08 en parquet (`id`, `created_at` UTC, `updated_at`, `headline`, `symbols`) y un informe de cobertura.
- **Ficheros:** `src/sentiment_portfolio/news.py`, `scripts/download_news.py`, `tests/test_news.py` (parseo y deduplicación con un JSON de fixture), `notebooks/01_news_eda.ipynb`.
- **Hecho cuando:**
  - El piloto de un mes mide el volumen real.
  - La descarga completa es reanudable (trozos de ticker × año) y deduplica por id.
  - Figuras del EDA:
    - Artículos por día, por ticker y año.
    - Histograma por hora ET (pre, intradía, post).
    - Distribución de `len(symbols)`.
    - Porcentaje de artículos con `updated_at − created_at` > 1 h.
  - Decisiones fijadas: fecha de inicio real y valor de K.
- **Riesgos:**
  - Cobertura fina en 2016 o en nombres como DUK o PG.
  - El etiquetado GOOG/GOOGL: se consultan ambos y se mapean a GOOGL.
  - Titulares editados tras publicarse (un look-ahead leve): se mide y se documenta.

### Fase 3: FinBERT, caché y alineación temporal
- **Objetivo:** una puntuación por artículo cacheada en disco y cada artículo asignado a su sesión con información, con tests.
- **Ficheros:** `src/sentiment_portfolio/{alignment,finbert}.py`, `scripts/score_news.py`, `tests/test_alignment.py`, `tests/test_finbert.py`.
- **Hecho cuando:**
  - Los tests de alineación pasan:
    - 15:59 ET → mismo día.
    - 16:00 y 16:01 → sesión siguiente.
    - Sábado → lunes.
    - Festivo → sesión siguiente.
    - Media sesión (p. ej. 2019-11-29) a las 14:00 → siguiente.
    - Frontera de cambio horario (DST) en marzo y noviembre.
  - Test de FinBERT: frases obvias puntúan con el signo correcto y el mapeo va por `id2label`.
  - Volver a ejecutar `score_news.py` no hace nada (100 % de aciertos de caché), con la revisión del modelo en la clave.
  - Throughput en CPU medido.
  - Revisión manual de 50 titulares al azar.
  - No uso Financial PhraseBank para validar: FinBERT se entrenó con él, así que la métrica no diría nada.
- **Riesgos:** unas 1–3 h de CPU [NV], ejecutadas una vez y reanudables por trozos; el boilerplate de Benzinga.

### Fase 4: Construcción y análisis de la señal (solo train)
- **Objetivo:** un panel diario de sentimiento por ticker, y saber si contiene información sobre rentabilidades futuras.
- **Ficheros:** `src/sentiment_portfolio/{signals,stats}.py`, `scripts/build_signal.py`, `tests/test_signal.py`, `notebooks/02_signal_analysis.ipynb`.
- **Hecho cuando:**
  - Pasa el **test de invariancia por truncado**: la señal hasta T es idéntica exista o no información después de T. Es el test genérico anti look-ahead.
  - Figura del IC (Spearman transversal) a 1, 5 y 20 días con su IC95 en train.
  - Correlación entre el sentimiento y la rentabilidad pasada (¿las noticias van detrás del precio?).
  - Tabla de cobertura por ticker.
- **Riesgos:** un IC ≈ 0. Seguimos igualmente, porque el proyecto va de hacerlo bien, pero las expectativas de la fase 5 quedan fijadas.

### Fase 5: Estrategia de tilt por sentimiento (train y validación)
- **Objetivo:** implementar la regla (b), elegir N y λ en train, confirmarla en validación y compararla con EW, SPY y los controles, con costes.
- **Ficheros:** `src/sentiment_portfolio/strategies.py`, extensión de `scripts/run_backtest.py`, `tests/test_strategies.py`, `notebooks/03_strategy.ipynb`.
- **Hecho cuando:**
  - Los tests pasan:
    - Los pesos suman 1 y respetan los límites.
    - λ=0 da exactamente EW.
  - Heatmap de Sharpe e IR frente a EW sobre el grid en train.
  - Tabla de validación con las métricas mínimas.
  - Análisis de sensibilidad a costes.
  - Distribución del placebo y su p-valor.
  - Control de momentum.
  - Bootstrap del ΔSharpe frente a EW.
  - Veredicto de la puerta de la fase 6.
- **Riesgos:** sobreajuste del grid (se mitiga reportándolo entero); validación de 2 años, muy ruidosa.

### Fase 6 (opcional, con puerta): PPO
- **Objetivo:** un agente PPO (SB3) con el vector de pesos como acción: entrena en train, se selecciona en validación, con al menos 5 semillas.
- **Ficheros:** `src/sentiment_portfolio/rl_env.py`, `scripts/train_ppo.py`, `tests/test_rl_env.py`.
- **Hecho cuando:**
  - Test: el env reproduce exactamente las rentabilidades del motor de backtest para los mismos pesos.
  - Test: la observación en t no contiene datos posteriores a t.
  - Tabla de mediana ± dispersión del Sharpe en validación frente al tilt.
- **Riesgos:** sobreajuste y varianza entre semillas. Las redes son pequeñas, así que la CPU basta.

### Fase 7: Evaluación final en test e informe
- **Objetivo:** abrir el test una sola vez con todas las estrategias congeladas y redactar el README con conclusiones honestas.
- **Ficheros:** `config/config.yaml` (congelado), `scripts/run_backtest.py --final`, `README.md`, `reports/figures/*`.
- **Hecho cuando:**
  - Existe el tag git `pre-test-freeze` **antes** de la ejecución final.
  - Tabla de test con las métricas mínimas para EW, SPY, tilt, controles y (PPO).
  - Figuras de equity y drawdown.
  - Test pareado y bootstrap del ΔSharpe.
  - El README tiene una sección "Qué NO demuestra esto".
- **Riesgos:** la tentación de iterar después de ver el test. El tag deja registrado el congelado, y cualquier cambio posterior se documenta como tal.

---

## 6. Decisiones (aprobadas el 2026-09-22 con la recomendación por defecto; se pueden revisar)
1. **Fuente:** Alpaca News. Tendrás que crear una cuenta paper gratuita y poner la clave en `.env`.
2. **Periodo y split:** 2016-01 → 2026-08, con train 2016–21, validación 2022–23 y test 2024-01 → 2026-08.
3. **Universo:** los 9 líderes sectoriales a 2016 según la regla ex-ante (lista de §3). Nada de elegir con los grandes de hoy.
4. **Timing:** la decisión se toma con las noticias hasta el cierre del último día de la semana y se ejecuta al cierre de la sesión siguiente.
5. **Texto para FinBERT:** solo el titular. Descartar los artículos con más de 3 símbolos.
6. **Costes:** 10 pb por dólar negociado, con sensibilidad 0, 5 y 25 pb.
7. **Tasa libre de riesgo para el Sharpe:** el T-bill de 3 meses (`^IRX`), en vez de 0.
8. **Controles extra (placebo y momentum):** incluirlos. Cuestan poco y son el núcleo de la honestidad estadística.
9. **Fase 6 (PPO):** fuera del MVP, solo si se pasa la puerta.
10. **Remoto en GitHub:** lo creas tú cuando quieras. Yo solo hago `git init` y commits locales.

## Verificación global
- Cada fase se cierra con su criterio de "hecho", `pytest` en verde y `ruff`.
- Desde un clon limpio, con la caché en `data/`, los scripts en orden (`download_prices`, `download_news`, `score_news`, `build_signal`, `run_backtest`) reproducen las figuras.
- Los tests anti look-ahead (alineación en la fase 3 y truncado en la fase 4) son obligatorios antes de cualquier resultado de estrategia.
