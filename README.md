# EDGE-Hetero: EDGE framework with Heterogeneous features

The **[EDGE](https://github.com/ds-jrg/EDGE)** framework is a novel framework for evaluating explanations from various node classifiers on knowledge graphs, utilizing advanced Graph Neural Networks and a range of evaluation metrics. It automates the evaluation process, aiming to quantitatively assess explainers and streamline evaluations with real-world datasets. This however does not cover heterogenous features of the graphs which are provided as input to these classifiers.

**EDGE-Hetero** justs simply seeks to add more flavor to the features used by the EDGE and report the findings of the behavior of the models and explainers with the added features.

## Logical Approaches
1. **EvoLearner:** [EvoLearner: Learning Description Logics with Evolutionary Algorithms](https://arxiv.org/abs/2111.04879)
2. **CELOE:**  [Class Expression Learning for Ontology Engineering](https://www.sciencedirect.com/science/article/pii/S1570826811000023)

The logical approaches in the EDGE framework, including EvoLearner and CELOE, were adapted from [OntoLearn](https://github.com/dice-group/Ontolearn).


## Sub-graph-based Approaches
For the subgraph approaches please refer to the [EDGE framework](https://github.com/ds-jrg/EDGE?tab=readme-ov-file#sub-graph-based-approaches)


## Datasets
The collection of benchmark datasets used on the EDGE framework are also supported

[A Collection of Benchmark Datasets for Systematic Evaluations of Machine Learning on the Semantic Web](https://link.springer.com/chapter/10.1007/978-3-319-46547-0_20)
1. [Mutag](https://pubmed.ncbi.nlm.nih.gov/1995902/)
2. [AIFB](https://link.springer.com/chapter/10.1007/978-3-540-76298-0_5)
3. [BGS](https://www.bgs.ac.uk/datasets/bgs-geology-625k-digmapgb/)


## Installation Guide for EDGE-Hetero
This install is pretty similar to setting up the EDGE framework 

### Step 1: Clone the EDGE Repository

First, clone the EDGE repository from GitHub using the following command:

```bash
git clone https://github.com/eugeneegbe/EDGE-Hetero.git
```

### Step 2: Install Conda

If you don't have Conda installed, download and install it from [Anaconda's official website](https://www.anaconda.com/products/individual).


### Step 3: Create the Conda Environment

```shell
conda create --name edge-hetero python=3.10 && conda activate edge-hetero
```

### Step 5: Install Dependencies

Navigate inside the EDGE directory using ( `cd EDGE ` ). Ensure you have a `requirements.txt` file in your project directory. To install the required dependencies, run:

```shell
pip install -r requirements.txt
```

This command will automatically install all the libraries and packages listed in your `requirements.txt` file. Based on your GPU / CPU devices, install the suitable version of DGL from official [DGL website](https://www.dgl.ai/pages/start.html). The experiments were carried out with the following version.
```shell
conda install -c dglteam/label/th23_cu121 dgl
```
At this level, edge-hetero is good to experiment with

## Dataset Preprocessing
For custom data preprocessing, please refer to the [EDGE framework](https://github.com/ds-jrg/EDGE?tab=readme-ov-file#dataset-preprocessing)


## Using and extending EDGE-Hetero

Same as the EDGE framework, to train models with specific models and/or datasets, use the `--train` flag along with `--model`, `--explainers`, `--datasets` and **`--describe`** flags as needed. See the examples below.

**Note:** As opposed to EDGE, Hetero uses RGAT as default model.

- Training all combination of explainers and datasets for 5 Runs with default RGAT
model, use the command:
  ```shell
  python main.py --train 
  ```

- Training specific explainers:
  ```shell
  python main.py --train  --explainers PGExplainer EvoLearner 
  ```

- Training models on specific datasets:
  ```shell
  python main.py --train --datasets mutag bgs
  ```

- Training models on specific datasets and describe the dataset:
  ```shell
  python main.py --train --datasets mutag bgs --describe True
  ```

- Combining specific explainers and datasets:
  ```shell
  python main.py --train --explainers SubGraphX CELOE --datasets aifb
  ```

- Train the RGCN model specifically:
  ```shell
  python main.py --train  --model RGCN 
  ```

- Print Results for specific using the model 
  ```shell
  python main.py --print_results --model RGCN
  ```

## RGCN Results
The RGCN results can be printed on the terminal using:
```shell
python main.py --print_results --model RGCN
```
If you just want to observe the results, <details><summary> Click me! </summary>

| Model | Dataset | Pred Accuracy | Pred Precision | Pred Recall | Pred F1 Score | Exp Accuracy | Exp Precision | Exp Recall | Exp F1 Score |
|---|---|---|---|---|---|---|---|---|---|
| CELOE | aifb | 0.722 | 0.647 | 0.733 | 0.688 | 0.744 | 0.694 | 0.745 | 0.718 |
| EvoLearner | aifb | 0.65 | 0.545 | 0.987 | 0.702 | 0.672 | 0.574 | 0.986 | 0.724 |
| PGExplainer | aifb | 0.667 | 0.605 | 0.68 | 0.634 | 0.689 | 0.647 | 0.696 | 0.666 |
| SubGraphX | aifb | 0.656 | 0.595 | 0.68 | 0.628 | 0.667 | 0.624 | 0.683 | 0.647 |
| CELOE | bgs | 0.517 | 0.409 | 0.9 | 0.563 | 0.531 | 0.436 | 0.889 | 0.583 |
| EvoLearner | bgs | 0.531 | 0.418 | 0.92 | 0.575 | 0.559 | 0.454 | 0.931 | 0.609 |
| PGExplainer | bgs | 0.538 | 0.38 | 0.54 | 0.441 | 0.566 | 0.442 | 0.592 | 0.497 |
| SubGraphX | bgs | 0.524 | 0.381 | 0.6 | 0.465 | 0.566 | 0.445 | 0.662 | 0.529 |
| CELOE | mutag | 0.703 | 0.718 | 0.92 | 0.804 | 0.632 | 0.617 | 0.897 | 0.726 |
| EvoLearner | mutag | 0.685 | 0.707 | 0.92 | 0.795 | 0.632 | 0.612 | 0.904 | 0.725 |
| PGExplainer | mutag | 0.456 | 0.663 | 0.373 | 0.475 | 0.691 | 0.889 | 0.587 | 0.681 |
| SubGraphX | mutag | 0.432 | 0.635 | 0.347 | 0.445 | 0.674 | 0.88 | 0.565 | 0.66 |

</details>
