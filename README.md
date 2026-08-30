# Gas Portfolio RL

Рабочий MVP синтетической Gymnasium-среды для управления портфелем из акции
`GasCo`, расчётного front-month фьючерса на газ, трёх защитных put-опционов и
cash. Финансовая логика разделена на симулятор рынка, портфельный accounting и
RL-обёртку.

## Установка и smoke test

```bash
python -m pip install -e .
pytest
python scripts/plot_prices.py --seed 42 --steps 504
```

График сохраняется в `results/reports/market_prices.png`.
Снимок наблюдаемой годовой forward curve сохраняется в
`results/reports/forward_curve.png`.

Интерактивный сценарий с графиками цен, запуском эпизода, dashboard портфеля,
пошаговым replay и примером подключения PPO/SAC находится в
`00_visualization_dashboard.ipynb`. В Google Colab ноутбук сам монтирует
`MyDrive`, находит `MyDrive/gas_portfolio_rl` и устанавливает пакет с нужными
зависимостями; результаты сохраняются обратно на Drive.

## Минимальный пример

```python
from gas_portfolio_rl import GasPortfolioEnv

env = GasPortfolioEnv()
observation, info = env.reset(seed=42)

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(info["portfolio_value"], info["pnl"], info["leverage"])
```

При включённых опционах действие среды имеет вид `[stock_weight,
futures_exposure, put_10d_coverage, put_21d_coverage, put_63d_coverage]`.
Суммарное put-покрытие ограничено единицей, опционы доступны только в long,
а среда проецирует действие в область, где cash покрывает премии, обеспечение
и резерв ликвидности. При `option.enabled=false` сохраняется прежнее
двумерное действие `[stock_weight, futures_exposure]`.

`info` и `env.trajectory` содержат cash, P&L, leverage, доли позиций, margin,
режим и издержки. Это готовый источник данных для статического или live-дисплея.

## Сезонность и term structure

Сезонный профиль задаётся датированными опорными точками в
`configs/market/default.yaml`: зимний пик приходится на конец декабря,
весенний минимум — на конец апреля, а летом формируется плато. При каждом
`reset()` амплитуда и фаза слегка рандомизируются.

Среда хранит календарную дату оценки, дату поставки front-month контракта и
12 контрактов forward curve. Агент получает циклические признаки обеих дат,
time-to-maturity, spot/front basis, наклоны, кривизну и winter-summer spread.
Истинные параметры сезонного профиля и рыночный режим остаются скрытыми.

GasCo имеет фиксированный dividend yield 3% годовых. Выплата происходит раз в
месяц 25-го числа в размере `annual_yield / 12`. На дату выплаты цена акции
корректируется вниз на dividend per share, поэтому покупка накануне не создаёт
безрисковую прибыль. Агент получает `time_to_dividend`, а momentum/hedge
baselines используют dividend-adjusted total return акции.

## Защитные put-опционы

Среда поддерживает три постоянно доступных семейства европейских cash-settled
put со сроками 10, 21 и 63 торговых дня. При выпуске strike фиксируется на 95%
текущей цены GasCo. После экспирации контракт рассчитывается по intrinsic value
и автоматически заменяется новым контрактом того же срока без физической
поставки.

Премия и Greeks рассчитываются по Black–Scholes. Implied volatility строится
только из наблюдаемой rolling volatility акции: в спокойном рынке term
structure восходящая, при стрессе она может инвертироваться. Покупка премии
обменивает cash на option asset и потому сама по себе не является убытком;
reward видит mark-to-market P&L, payoff, bid/ask spread и комиссию без двойного
учёта. Лимиты на стоимость опционов и годовой premium budget вынесены в
`configs/environment/default.yaml`.

## Reward и риск-бюджет

Reward v3 использует чистую доходность капитала сверх доходности cash. Прибыль
явно раскладывается на stock/futures/option P&L, дивиденды, cash interest и
комиссии, но входит в reward ровно один раз. Фьючерс и опционы получают общий
signed hedge-effectiveness signal за изменение loss-averse utility относительно
того же stock/cash портфеля без hedge-инструментов. Поэтому кризисная защита
получает bonus, а carry hedge в положительные дни — симметричный cost. Отдельно штрафуются
прирост drawdown, turnover и превышение безопасной загрузки маржи. Margin call
и банкротство завершают эпизод с крупным накопительным штрафом.

Перед PPO reward проверяется на статической сетке допустимых портфелей. При
включённых опционах она содержит 54 политики: исходные 15 stock/futures
комбинаций, 50%/100% put ladder для каждого futures exposure и полное покрытие
каждым отдельным сроком. При выключенных опционах сохраняется legacy-grid из
15 политик. Опционные grid-политики восстанавливают целевое coverage через тот
же `decision_interval`, что и PPO, поэтому контракты продолжают покупаться после
экспирации. Основной gate
требует alignment с внешней utility и отсутствия фатальных событий. Положение
статического optimum на границе и использование futures сохраняются отдельно
как diagnostics: статический hedge не принуждается выигрывать вопреки данным.
Обучение из `03` допускается только при `passes_reward_gate=true`.

Все пороги и коэффициенты вынесены в `configs/reward/default.yaml`:

```python
from gas_portfolio_rl import AppConfig, GasPortfolioEnv

config = AppConfig.from_yaml(
    "configs/market/default.yaml",
    "configs/environment/default.yaml",
    "configs/reward/default.yaml",
)
env = GasPortfolioEnv(config)
```

## Baseline-стратегии и live-monitor

Основные стратегии:

- `RelativeMomentumBaseline` сравнивает dividend-adjusted risk-adjusted
  momentum акции и фьючерса и размещает 50% в более сильном инструменте;
- `RollingHedgeBaseline` держит 50% в GasCo и рассчитывает короткий фьючерсный
  hedge по rolling minimum-variance beta.
- `SignedRollingHedgeBaseline` использует signed minimum-variance beta: short
  future при положительной covariance и long future при отрицательной. Для
  устойчивости применяются shrinkage, сглаживание, correlation threshold,
  deadband и симметричный exposure limit.
- `ProtectivePutBaseline` держит заданную долю GasCo и распределяет защитное
  покрытие между 10/21/63-дневными put. В evaluation доступны отдельные
  short/medium/long варианты и диверсифицированная put ladder.

Параметры находятся в `configs/baselines/default.yaml`. Обе стратегии
ребалансируются раз в пять торговых дней. Между ребалансировками позиции
удерживаются без микросделок.

Интерактивный Colab-сценарий находится в `01_baselines_live_monitor.ipynb`.
Дисплей обновляет P&L, cash, комиссии, дивиденды, распределение портфеля,
рыночные цены, risk/margin-метрики и ленту сделок. Быстрый запуск из терминала:

```bash
python scripts/run_baseline_monitor.py --strategy momentum --steps 252
python scripts/run_baseline_monitor.py --strategy hedge --steps 252
python scripts/run_baseline_monitor.py --strategy signed-hedge --steps 252
```

Live-режим использует одну постоянную figure и обновляет её через Jupyter
`DisplayHandle`, не вызывая `clear_output()`. По умолчанию показывается
скользящее окно последних 80 дней с refresh каждые 2 шага; параметры меняются
через `window_size`/`refresh_every` или CLI-флаги `--window-size` и
`--refresh-every`. Сохранённый PNG всегда содержит полную историю.

Для запуска без live-обновлений добавьте `--no-live`. Итоговый dashboard
сохраняется в `results/reports/`. Будущая Stable-Baselines3 модель подключается
к тому же runner и дисплею через `ModelStrategy(model, name="ppo")`.

## Калибровка и evaluation gate перед PPO

`02_calibration_and_evaluation.ipynb` реализует обязательный этап до обучения:

- simulation-based калибровка параметров рынка по волатильностям, корреляции и
  доле кризисного режима;
- непересекающиеся seed-диапазоны calibration/train/validation/test/stress;
- Gymnasium, determinism и finite-value preflight checks;
- парное сравнение cash, stock-only, fixed hedge, relative momentum и rolling
  hedge на одинаковых рыночных траекториях;
- bootstrap confidence intervals, Sharpe, Sortino, max drawdown, CVaR, Calmar,
  turnover, комиссии, дивиденды, margin calls и банкротства;
- отдельные calm и crisis-stress сценарии;
- диагностика соответствия reward внешней risk-adjusted utility.

Calibration gate требует одновременно приемлемого общего objective и
прохождения индивидуальных допусков для каждой volatility, correlation и доли
кризиса. Terminal reward за margin call включает факт события, потерянную долю
капитала и оставшуюся долю горизонта, поэтому ранняя ликвидация не сокращает
накопительный штраф искусственно.

Быстрая сквозная проверка запускается командой:

```bash
python scripts/run_pre_ppo_pipeline.py --quick
```

Полный запуск без `--quick` сохраняет воспроизводимые артефакты в
`results/pre_ppo/`. Калиброванные параметры сохраняются отдельно и не
перезаписывают исходный `configs/market/default.yaml`. До появления реальных
рыночных данных calibration targets являются явно задокументированными
моделирующими допущениями, а не эмпирическими оценками.

Все корневые notebooks используют `notebook_bootstrap.py`. При запуске в
Colab bootstrap монтирует проект из `MyDrive/gas_portfolio_rl`, устанавливает
зависимости и явно добавляет `PROJECT_ROOT/src` первым в `sys.path`. После
перезапуска Colab runtime следует выполнить notebook через `Runtime -> Run all`;
успешная ячейка печатает точный путь `Package imported from: ...`.

## PPO: обучение, model selection и финальная оценка

Полный PPO-пайплайн реализован в `03_train_ppo.ipynb` и
`04_evaluate_ppo.ipynb`. Обучение использует Stable-Baselines3, четыре
параллельные среды, отдельные диапазоны train/validation seed и `VecNormalize`
для observations/reward. Resolver допускает к обучению только market-config,
у которого одновременно прошли общий и покомпонентный calibration gate.

Action schema PPO v4 принимает решение раз в пять рыночных дней. Эксперимент
`v5_full` сохраняет это совместимое пространство действий, но использует новый
risk budget reward и более стабильные параметры оптимизации. Шестимерный выход policy
задаёт stock, futures, общее put-покрытие и три logits распределения покрытия
по срокам. Он преобразуется в пятимерное действие среды; softmax гарантирует
корректную сумму tenor allocations. Stock weight ограничен `[0.25, 0.75]`, а
модуль futures exposure — `min(stock_weight, 0.50)`. Общее put-покрытие не
опускается ниже 25%, переход к новому распределению сроков сглаживается с
коэффициентом 0.35, а option deadband 0.06 подавляет малые сделки. Две из четырёх
параллельных training-сред используют crisis-stress рынок; метка режима агенту
не сообщается. Полный run содержит 600 000 PPO decisions, то есть примерно
3 млн внутренних дневных переходов.

Validation callback через каждые 20 000 решений сравнивает checkpoints на
фиксированных base и crisis-validation seed. Score учитывает base/crisis
reward, margin calls, bankruptcies, недостаточное участие в акции, чрезмерный
turnover и вырожденную политику, которая почти всегда остаётся в cash. В
crisis-сценарии score также штрафуется за превышение drawdown статического
`minimum_stock_put_ladder`; baseline рассчитывается один раз на тех же seed.
Вместе с лучшей моделью обязательно сохраняется соответствующее состояние
`VecNormalize`; загружать их нужно парой.

Команды для проверки и полного запуска:

```bash
# короткая проверка всего train-пайплайна
python scripts/train_ppo.py --smoke

# полное обучение PPO v5_full
python scripts/train_ppo.py

# validation: разрешено использовать при выборе модели
python scripts/evaluate_ppo.py --run-dir results/ppo/v5_full/run_YYYYMMDD_HHMMSS

# untouched test: один раз для окончательно выбранной модели
python scripts/evaluate_ppo.py --run-dir results/ppo/v5_full/run_YYYYMMDD_HHMMSS --test
```

Основной профиль находится в `configs/models/ppo_v5_full.yaml`; профили
`v5_control`, `v5_long_horizon` и `v5_stable` предназначены для последовательной
ablation-проверки. Артефакты одного
smoke-run занимают около 2 MiB. Поэтому ограничения Colab в 12 GiB RAM и Google
Drive менее 15 GiB достаточны; для экономии места следует хранить best/final
bundle и лишь несколько checkpoints, а не сериализовать replay buffer (у PPO
его нет).

Validation notebook показывает cumulative terminal net PnL в денежных единицах
отдельными столбчатыми диаграммами: текущие baselines/PPO и исторические PPO
версии. Каждая validation/test оценка регистрируется в
`results/ppo/registry/ppo_history.csv`; v4 reference уже сохранён там. Реестр
содержит return, PnL, volatility, tail/drawdown, allocation, option и reward
decomposition метрики, а leaderboard ранжирует только полные трёхсценарные
запуски. Новые summary содержат
поле `net_pnl_mean`; старые результаты отображаются через эквивалент
`total_return_mean * initial_cash`.

Acceptance gate требует отсутствия bankruptcy, margin-call rate не выше 1%,
return/reward выше cash в base и calm, crisis-защиту относительно signed hedge
и cash, минимум 10% средней stock allocation и turnover не выше 20. Кроме
того, robust reward PPO обязан быть выше обнаруженного grid benchmark
опционного benchmark `minimum_stock_put_ladder`.
`--smoke` проверяет инфраструктуру, но не обязан проходить quality gate.

`03_train_ppo.ipynb` обновляет живой training dashboard после каждой validation:
rolling reward/episode length, base/crisis reward, return/drawdown, fatal-event
rates, stock weight и turnover. `04_evaluate_ppo.ipynb` строит сравнительные
графики PPO/cash/signed hedge и сохраняет их вместе с CSV, acceptance JSON,
`versioned_net_pnl.png` и обновлённым историческим реестром. Конфигурация
валидации всегда восстанавливается из `run_metadata.json`, поэтому старый
checkpoint не оценивается с reward или нормализацией новой версии.
