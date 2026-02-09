# Two failure modes of deep transformers and how to avoid them  
## A unified theory of signal propagation at initialisation

This repository contains the notebooks used to reproduce the figures in the paper:

> **Two failure modes of deep transformers and how to avoid them:  
> A unified theory of signal propagation at initialisation**  
> Alessio Giorlandino & Sebastian Goldt  
> *International Conference on Learning Representations (ICLR) 2026*

📄 Paper (OpenReview): https://openreview.net/forum?id=utSqpxQHXq  
📄 arXiv: https://arxiv.org/abs/2505.24333  

---

## 📁 Files and Figure Mapping

This repository is organized around **Jupyter notebooks used to reproduce the main figures and results** of the paper, together with a small custom PyTorch implementation of BERT-style encoders with tunable residual connections.

### 📓 Notebooks

- **`deep_prop_encoder.ipynb`**  
  Reproduces **Result 1 / Figure 1**: comparison between theoretical predictions and experiments for signal propagation in encoder models.

- **`gradients.ipynb`**  
  Reproduces **Result 2 / Figure 3**: analysis of gradient behavior at initialization.

- **`Gain-controlled-propagation.ipynb`**  
  Implements the theoretical propagation analysis of **Figure 4**, comparing:
  - post-LN vs pre-LN  
  - vanilla vs gain-controlled attention  

- **`deep_prop_decoder.ipynb`**  
  Reproduces **Figure 12**: comparison between theory and experiments for propagation of late-sequence tokens in decoder models.

Each notebook is self-contained and can be run top-to-bottom to regenerate the corresponding plots.

---

### 🧩 Custom BERT implementation (PyTorch)

These files implement a **customized BERT-style encoder with tunable residual connections**, used in the numerical experiments:

- `custom_bert_model.py`  
- `custom_bert_encoder.py`  
- `custom_bert_layer.py`  
- `custom_bert_layers.py`

They are lightweight modifications of standard Transformer/BERT components, designed specifically to expose residual scaling parameters required for the theoretical comparisons.

---

### 🗃 Tokenizer utilities

Used only for small-scale experiments:

- `tiny_tokenizer.py`  
- `tiny-tokenizer.json`

---

## 📎 Citation

If you find this work useful, please cite:

```bibtex
@misc{giorlandino2025failuremodesdeeptransformers,
      title={Two failure modes of deep transformers and how to avoid them: a unified theory of signal propagation at initialisation}, 
      author={Alessio Giorlandino and Sebastian Goldt},
      year={2025},
      eprint={2505.24333},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2505.24333}, 
}
