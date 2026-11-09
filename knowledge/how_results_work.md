# How VisionGuard results work

## What a verdict is
VisionGuard looks at one image and estimates whether it was made by an AI image generator. The answer has three parts: a verdict (AI-generated or Real), a probability of AI and a confidence. A verdict is an estimate from a small model. It is not proof and not a forensic finding.

## Probability of AI
The model outputs a number between 0 and 1 called the probability of AI. A value of 0.80 means the model scores the image as strongly AI-like. This number is a score, not a measured chance. It is not calibrated, so an image scored 0.80 is not guaranteed to be AI-generated 80 times out of 100.

## The threshold
An image is labelled AI-generated when its probability of AI is at or above the threshold, which is 0.48 for model version 2. Below the threshold the label is Real. The threshold was chosen on validation images so that the model catches most AI images, accepting that some real images are flagged by mistake.

## Confidence
The confidence shown next to a verdict is the model's score for the label it chose. For an AI-generated verdict it equals the probability of AI. For a Real verdict it equals one minus the probability of AI. A confidence of 90 percent means a strong score, not a 90 percent chance of being right. A confidence near 50 percent means the model is unsure.

## Why results can be wrong
The model was trained on about 1,000 images, so it is a small model. On 147 held-out test images it labelled about 79 percent correctly. It reacts to style, such as saturated colours, painterly texture and collage layouts, more than to real signs of AI generation. Human-made paintings and collages are sometimes flagged as AI, and photorealistic AI images are sometimes called Real.

## Borderline results
Scores close to the threshold, for example a probability of AI between 0.35 and 0.65, are borderline. For borderline results, treat the verdict as a reason to look closer, and do not rely on it.