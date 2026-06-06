# BrainTopologyLLM

Топологическая оптимизация нейросетей на основе кривизны Риччи.

## Идея

Мы применяем математический аппарат потока Риччи (доказательство гипотезы Пуанкаре) к графам активаций нейронов. Узлы с низкой топологической важностью можно удалить без потери точности.

## Результаты (MNIST CNN)

Сравнение методов pruning на 186 нейронах:

- Ricci topology: 30% нейронов удалено, точность 90.56% (baseline 90.98%)
- Weight magnitude: катастрофический провал после 10% (точность падает до 9.80%)
- Random: закономерно деградирует с ростом pruning

Ricci превосходит random на +18.4% в среднем.
Ricci превосходит magnitude на +76% при 10%+ pruning.
При 10% pruning точность улучшается на +0.33% (эффект регуляризации).

График: results/pruning_comparison.png

## Структура проекта

brain_topology_llm/
  src/
    run_ricci.py          - базовый пайплайн: граф -> Ricci -> pruning
    compare_pruning.py    - сравнение трёх методов
  results/                - графики
  README.md
  .gitignore

## Быстрый старт

python3.10 -m venv venv_brain
source venv_brain/bin/activate
pip install torch torchvision networkx GraphRicciCurvature numpy matplotlib
python src/compare_pruning.py

## Следующие шаги

- Multi-seed validation
- CIFAR-10 / ResNet
- Перенос на LLM (GPT-2)
- Сжатие промптов

## Лицензия

MIT
