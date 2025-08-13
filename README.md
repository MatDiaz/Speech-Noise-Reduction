# Speech-Noise-Reduction

Este proyecto implementa un sistema de reducción de ruido en señales de voz utilizando redes neuronales en PyTorch.

## Descripción
El sistema toma como entrada el espectro de magnitudes de audio ruidoso y predice el espectro de magnitudes del audio limpio.  
Posteriormente, el audio sin ruido se reconstruye en el dominio temporal utilizando la fase del audio ruidoso.

## Flujo del Proceso
1. **Carga y preprocesamiento de datos**
   - Descarga de un corpus de voz.
   - Aumento de datos añadiendo ruido blanco a las señales limpias.
   - Cálculo del STFT y obtención de magnitudes.
   - Segmentación en ventanas (`num_segments`).
   - Normalización de datos.

2. **División de datos**
   - Separación en conjuntos de entrenamiento y validación.

3. **Entrenamiento**
   - Red neuronal totalmente conectada (`DenoiseNet`) con capas lineales y batch normalization.
   - Función de pérdida: **MSELoss**.
   - Optimización con Adam.

4. **Evaluación**
   - Reconstrucción del audio denoised mediante ISTFT.
   - Métricas sugeridas: **PESQ**, **STOI**, **SDR** (requieren librerías externas).

5. **Exportación**
   - Guardado del modelo y parámetros adicionales con `torch.save`.

## Requisitos
- Python 3.8+
- PyTorch
- Torchaudio
- IPython (para reproducir audio en notebooks)
- Librerías opcionales: `pesq`, `torchmetrics`, `pystoi`

## Uso
```python
checkpoint = torch.load("modelo.pth")
model = checkpoint["model"]
model_params = checkpoint["model_params"]

output = denoise_audio(model, noisy_audio, model_params, ...)
