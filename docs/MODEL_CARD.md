# Model card: VisionGuard champion (version 2)

## Task and labels
- Binary image classification. 0 = Real, 1 = AI-generated (positive class).
- Output: label, probability of AI (prob_ai), confidence, threshold, model name and version.
- Decision threshold: 0.48, chosen on the validation set. No bounding boxes.

## Data
- Kaggle "AI Generated Images vs Real Images": 975 images, 1 exact duplicate removed.
- Fixed seeded split 70/15/15 = 681 train / 146 validation / 147 test, versioned with DVC (DagsHub).
- User uploads are not used for training.

## Model
- MobileNetV2 (ImageNet weights) + global average pooling + dropout 0.2 + 1 sigmoid unit.
- Fine-tuned: last 30 backbone layers unfrozen, learning rate 1e-4 (assignment run r4).
- Validation ROC-AUC 0.916 (selection metric). Test results below.

## Preprocessing (identical in training and serving)
1. Decode to RGB (3 channels), pixel values 0 to 255.
2. Resize to 224 x 224 with bilinear interpolation, no anti-aliasing, half-pixel centres
   (matches tf.image.resize). The aspect ratio is NOT preserved (image is stretched).
3. Scaling x / 127.5 - 1 happens inside the model.
- The API implements step 2 in NumPy (api/preprocess.py) so serving needs no TensorFlow.
  A unit test checks it against tf.image.resize.

## Test-set results (147 images, threshold 0.48, API preprocessing)
| Model | Accuracy | ROC-AUC | Labels changed vs Keras | Size | Latency (1 CPU thread, laptop) |
|---|---|---|---|---|---|
| Keras reference | 0.796 | 0.875 | 0 | n/a | n/a |
| Float TFLite (deployed) | 0.789 | 0.873 | 1 | 8.89 MB | 29 ms |
| Quantized TFLite (not adopted) | 0.776 | 0.874 | 3 | 2.53 MB | 135 ms |

The old Pillow-based resize changed 7 of 147 labels (max probability gap 0.44); the new one changes 1
(max gap 0.035). Quantization was tested and rejected: 2 extra label changes and about 4.6 times slower on CPU.

## Limits
- Small data: 147 test images means roughly +/- 7 percentage points of uncertainty on accuracy.
- The model appears to use style (saturation, painterly texture, collage layout) more than evidence of
  synthesis; paintings and collages in the real class are often flagged as AI.
- Image size, file size and format differ between the classes (possible shortcuts).
- The output is a model score, not proof. Treat it as an indicator for review.
