import os

import uvicorn

from app import createApp

app = createApp()

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "5000")), log_level="info")