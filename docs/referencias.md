# Referências

## Keyword spotting (base do projeto)

- ZHANG, Yundong et al. **Hello Edge: Keyword Spotting on Microcontrollers.** arXiv:1711.07128, 2017. https://arxiv.org/abs/1711.07128 — DS-CNN, orçamento de memória/MACs para a borda.
- KIM, Byeonggeun et al. **Broadcasted Residual Learning for Efficient Keyword Spotting (BC-ResNet).** arXiv:2106.04140, 2021. https://arxiv.org/abs/2106.04140
- BERG, Axel et al. **Keyword Transformer: A Self-Attention Model for Keyword Spotting (KWT).** arXiv:2104.00769, 2021. https://arxiv.org/abs/2104.00769
- TANG, Raphael; LIN, Jimmy. **Deep Residual Learning for Small-Footprint Keyword Spotting** (res8/res15). ICASSP 2018. https://arxiv.org/abs/1710.10361
- WARDEN, Pete. **Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.** arXiv:1804.03209, 2018. https://arxiv.org/abs/1804.03209
- BETTER STACK. **Speech Recognition Without the Internet Now Fits on a $10 Chip (Whistle).** YouTube, 2026. https://youtu.be/bavO_V80ohM

## Arquiteturas vistas na disciplina

- BENGIO, Y. et al. **A Neural Probabilistic Language Model.** JMLR, 2003 — MLP.
- KRIZHEVSKY, A. et al. **ImageNet Classification with Deep Convolutional Neural Networks.** NeurIPS 2012 — CNN.
- HE, K. et al. **Deep Residual Learning for Image Recognition.** CVPR 2016 — ResNet.
- TAN, M.; LE, Q. **EfficientNet: Rethinking Model Scaling for CNNs.** ICML 2019 — escalonamento e convoluções separáveis.
- RONNEBERGER, O. et al. **U-Net.** MICCAI 2015.
- VASWANI, A. et al. **Attention Is All You Need.** NeurIPS 2017 — Transformer.
- DEVLIN, J. et al. **BERT.** arXiv:1810.04805, 2018.
- RADFORD, A. et al. **Language Models are Unsupervised Multitask Learners** (GPT-2), 2019.
- RADFORD, A. et al. **Learning Transferable Visual Models From Natural Language Supervision** (CLIP). ICML 2021.
- LECUN, Y.; BENGIO, Y.; HINTON, G. **Deep Learning.** Nature 521, 2015.

## Datasets

- Google Speech Commands v2 — https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- DEMAND (ruído ambiente) — https://zenodo.org/records/1227121
- AudioMNIST — https://github.com/soerenab/AudioMNIST
- Multilingual Spoken Words Corpus (pt) — https://mswc.mlcommons-storage.org/audio/pt.tar.gz

## Achados da POC ([kws-borda-poc](https://github.com/gpeuzinCosta/kws-borda-poc))

| Modelo | Acurácia (35 cl.) | Params | MACs | Rede (ms) |
|---|---|---|---|---|
| MLP | 68,3% | 265 mil | 0,26 M | 0,10 |
| CNN 1D | 91,7% | 47 mil | 1,9 M | 0,36 |
| ResNet (res8) | 94,5% | 112 mil | 37,2 M | 1,69 |
| DS-CNN | 93,1% | 25 mil | 21,7 M | 1,40 |
| Atenção (KWT-tiny) | 89,3% | 79 mil | 9,6 M | 0,52 |

- O MFCC levou 0,38 ms com torchaudio e 1,91 ms com librosa: a resposta da hipótese (b) depende da biblioteca.
- O Zenodo 1227121 é **ruído** (DEMAND), não fala, e serve para o teste de robustez.
- O AudioMNIST varia **sala e sotaque**, não o microfone.
- No MSWC-pt, o modelo usado direto fica perto do acaso, mas com cabeça linear e 20 exemplos por palavra chega a ~85%.
- Os erros são fonéticos: forward→four, tree↔three, go↔no, off→up.
