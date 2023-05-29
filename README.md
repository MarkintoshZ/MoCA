# Reproducing Momentum Calibration for Text Generation Paper

Part of onboarding task for Forward Data Lab Summer 2023, I reproduced the current state-of-the-art abstractive text summerication method: [Momentum Calibration for Text Generation](https://arxiv.org/abs/2212.04257) (MoCA) using huggingface and pytorch. See [Final Report](Summarization%20Research%20Project.pdf) for this task.

### Results:
| | Rouge 1 | Rouge 2 | Rouge L |
| - | -| - | - |
| Vanilla Finetuned Baseline in the MoCa Paper | 44.22 | 21.22 | 41.01 |
| MoCa Paper | 48.88 | 24.94 | 45.76 |
| Vanilla Finetuned Baseline in the BART paper | 40.95 | 20.81 | 40.04 |
| My Implementation | 46.04 | 22.61 | 42.93 |

### Running the code
- Set/Change the environment variable for selecting GPUs to use.
- Run ```python MoCa.py ``` to apply MoCA on model.
- Run ```evaluation.ipynb``` to evaluate model on ROUGE score.