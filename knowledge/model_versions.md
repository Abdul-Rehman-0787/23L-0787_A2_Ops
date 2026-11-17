# Model versions

## What the model versions are
VisionGuard stores the model name and version with every result. Version 2 is the full-precision model: the champion from the experiment registry, with the numbers described in the model card. Version 2-quant is a compressed copy of the same model. Its weights are stored with fewer digits, which makes the file about 3.5 times smaller (2.5 MB instead of 8.9 MB). Both versions look at the same pixels in the same way. They are the same trained model in two file formats, not two different training runs.

## Why two versions can give different results
Compression changes the numbers inside the model very slightly, so the probability of AI changes slightly too. On the 147 test images the compressed copy differed from the original by 0.02 on average (2 percentage points) and by up to 0.16 (16 points) on the worst image. When an image's score sits close to the threshold of 0.48, a change of a few points can move it to the other side, and the verdict flips between AI-generated and Real. Images far from the threshold keep the same verdict.

## Which version should I trust
Neither version is proof. Version 2 is the reference: its test accuracy was about 79 percent, and the numbers in the model card describe it. Version 2-quant scored about 78 percent on the same test images and changed 3 labels compared with the original. When two versions disagree on the same image, the model is effectively unsure about that image. Treat it as an unclear case that needs a human check, and look at how close the two probabilities are to 0.48.

## How to tell that two results are the same image
Every result keeps a SHA-256 fingerprint of the file. Two results with the same fingerprint are the same file, byte for byte. A resized or re-saved copy gets a different fingerprint and counts as a different image.