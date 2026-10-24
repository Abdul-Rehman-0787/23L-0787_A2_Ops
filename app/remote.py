import logging
import time

import requests

log = logging.getLogger("visionguard")


class RemotePredictor:
    """Same predict() as the local Predictor, but the model runs in the cloud."""

    def __init__(self, url, api_key, timeout=90, retries=2):
        if not url or not api_key:
            raise RuntimeError("API_URL and API_KEY must be set when INFERENCE_MODE=remote")
        self.endpoint = url.rstrip("/") + "/predict"
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    def predict(self, data: bytes):
        last = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(self.endpoint,
                                  headers={"X-API-Key": self.api_key},
                                  files={"file": ("image.jpg", data, "image/jpeg")},
                                  timeout=self.timeout)
            except requests.RequestException as e:      # network problem or timeout
                last = e
            else:
                if r.status_code not in (502, 503, 504):  # 502-504: free instance waking up
                    if r.status_code != 200:
                        raise RuntimeError(f"Cloud API error {r.status_code}: {r.text[:200]}")
                    j = r.json()
                    return {"label": j["label"], "confidence": j["confidence"],
                            "prob_ai": j["prob_ai"], "threshold": j["threshold"],
                            "model_name": j.get("model_name"),
                            "model_version": j.get("model_version")}
                last = RuntimeError(f"Cloud API warming up (HTTP {r.status_code})")
            log.warning("Cloud call failed (attempt %d/%d): %s", attempt + 1, self.retries + 1, last)
            if attempt < self.retries:
                time.sleep(5)
        raise RuntimeError(f"Cloud API unreachable: {last}")