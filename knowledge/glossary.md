# VisionGuard glossary

## AI-generated image
An image created by a generative model, for example a diffusion model, from a text prompt rather than taken by a camera or drawn by a person.

## Classifier
A program that puts an input into one of a few classes. VisionGuard has two classes: Real and AI-generated.

## False alarm
A false alarm, also called a false positive, is a real image that the model labels AI-generated.

## Miss
A miss, also called a false negative, is an AI-generated image that the model labels Real.

## Precision
Of all the images the model labels AI-generated, precision is the share that really are AI-generated.

## Recall
Of all the AI-generated images, recall is the share that the model finds.

## ROC-AUC
A number from 0.5 to 1 that shows how well the model's score ranks AI images above real images across all thresholds. 0.5 is guessing and 1.0 is perfect. The model scored about 0.87 on the test images.

## SHA-256 hash
A fingerprint of a file. The same file always gives the same hash, and a different file gives a different one. VisionGuard saves the hash instead of the image.

## One-time code
A 6-digit code emailed to you at login. It works once and expires after 5 minutes.