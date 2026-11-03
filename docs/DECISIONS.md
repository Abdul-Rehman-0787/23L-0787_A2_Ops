# VisionGuard Platform: decisions

- Use case: classify an image as AI-generated or real (not defect or object detection).
- Output: label + score + hash + timestamp. No bounding boxes (classification only).
- Dataset: Kaggle "AI Generated Images vs Real Images". Licence: TODO, read the Kaggle page and paste it here.
- Model: MobileNetV2 fine-tuned, converted to TensorFlow Lite.
- OTP delivery: email. Local development uses a test inbox (Mailpit); hosted channel decided at deployment.
- External processing: only Artist's text questions may go to Groq in the cloud; images are not sent to external LLMs.
- Budget: free tiers only. No paid service without approval.
- MVP priority: login + upload/inference + saved results + basic Artist answer.