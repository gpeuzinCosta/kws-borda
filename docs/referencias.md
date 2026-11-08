# Referências

## Keyword spotting
- ZHANG, Yundong et al. **Hello Edge: Keyword Spotting on Microcontrollers.** arXiv:1711.07128, 2017. https://arxiv.org/abs/1711.07128 — DS-CNN, orçamento de memória/MACs para a borda.
- KIM, Byeonggeun et al. **Broadcasted Residual Learning for Efficient Keyword Spotting (BC-ResNet).** arXiv:2106.04140, 2021. https://arxiv.org/abs/2106.04140
- BERG, Axel et al. **Keyword Transformer: A Self-Attention Model for Keyword Spotting (KWT).** arXiv:2104.00769, 2021. https://arxiv.org/abs/2104.00769
- TANG, Raphael; LIN, Jimmy. **Deep Residual Learning for Small-Footprint Keyword Spotting** (res8/res15). ICASSP 2018. https://arxiv.org/abs/1710.10361
- WARDEN, Pete. **Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.** arXiv:1804.03209, 2018. https://arxiv.org/abs/1804.03209
- BETTER STACK. **Speech Recognition Without the Internet Now Fits on a $10 Chip (Whistle).** YouTube, 2026. https://youtu.be/bavO_V80ohM

## Microcontroladores (§6.1 do relatório)

- GOBIESKI, Graham; LUCIA, Brandon; BECKMANN, Nathan. **Intelligence Beyond the Edge: Inference on Intermittent Embedded Systems.** ASPLOS 2019. https://arxiv.org/abs/1810.07751 — redes neurais no MSP430FR5994.
- WARDEN, Pete; SITUNAYAKE, Daniel. **TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers.** O'Reilly, 2019 — exemplo `micro_speech` no Arduino Nano 33 BLE Sense.
- TENSORFLOW. **How-to Get Started with Machine Learning on Arduino.** 2019. https://blog.tensorflow.org/2019/11/how-to-get-started-with-machine.html
- MICROCHIP. **ATmega328P Datasheet** (Arduino Uno: 32 KB Flash). https://www.microchip.com/en-us/product/atmega328p
- TEXAS INSTRUMENTS. **MSP430FR599x, MSP430FR596x Mixed-Signal Microcontrollers** (256 KB FRAM). https://www.ti.com/product/MSP430FR5994
- NORDIC SEMICONDUCTOR. **nRF52840 Product Specification** (1 MB Flash). https://www.nordicsemi.com/Products/nRF52840
- ESPRESSIF. **ESP32-WROOM-32 Datasheet** (4 MB Flash SPI). https://www.espressif.com/sites/default/files/documentation/esp32-wroom-32_datasheet_en.pdf

## Arquiteturas vistas na disciplina

- BENGIO, Y. et al. **A Neural Probabilistic Language Model.** JMLR, 2003 — MLP.
- KRIZHEVSKY, A. et al. **ImageNet Classification with Deep Convolutional Neural Networks.** NeurIPS 2012 — CNN.
- HE, K. et al. **Deep Residual Learning for Image Recognition.** CVPR 2016 — ResNet.
- TAN, M.; LE, Q. **EfficientNet: Rethinking Model Scaling for CNNs.** ICML 2019 — convoluções separáveis (base da DS-CNN).
- VASWANI, A. et al. **Attention Is All You Need.** NeurIPS 2017 — Transformer (base do KWT).

## Datasets

- Google Speech Commands v2 — https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- DEMAND (ruído ambiente) — https://zenodo.org/records/1227121
- AudioMNIST — https://github.com/soerenab/AudioMNIST
- Multilingual Spoken Words Corpus (pt) — https://mswc.mlcommons-storage.org/audio/pt.tar.gz
