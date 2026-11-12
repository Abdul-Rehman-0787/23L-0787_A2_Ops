**3 commands, or 1 if you use the start script below.** Everything runs on your laptop.

## What runs where

| Part | How it starts |
|---|---|
| Redis, Mailpit (fake inbox), Neo4j | **Command 1:** `docker compose up -d redis mailpit neo4j` |
| Celery worker (emails and graph updates) | **Command 2**, in its own window: `celery -A app.tasks:celeryApp worker --pool=solo --loglevel=info` |
| Web app, which also loads the model | **Command 3**, in its own window: `python run.py` |
| Ollama (Artist's language model) | Starts by itself with Windows (llama icon in the tray). No command. |
| Upload | No command. You use the browser at http://localhost:5000. |

**The model:** with `INFERENCE_MODE=local` in `.env`, the web app loads `models\champion.keras` itself, so there's nothing separate to start. It takes about 10 to 20 seconds on startup and works offline. Use `local` on the `platform` branch. The `remote` mode calls your Render API, which deploys from `main`, so it still has the old Pillow resize from before Day 2 and gives slightly different answers.

## One command instead: `dev.ps1`
Create `dev.ps1` in the project root:
```powershell
# Starts everything VisionGuard needs on this laptop.
$project = $PSScriptRoot
Set-Location $project
$python = Join-Path $project ".venv\Scripts\python.exe"

Write-Host "1/4 Starting Redis, Mailpit and Neo4j (Docker)..."
docker compose up -d redis mailpit neo4j
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker could not start them. Is Docker Desktop running, and is NEO4J_PASSWORD set in .env?"
    exit 1
}

Write-Host "2/4 Checking Ollama..."
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 | Out-Null
    Write-Host "Ollama is running."
} catch {
    Write-Host "Ollama was not running. Starting it..."
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

Write-Host "3/4 Starting the worker in a new window..."
$workerCommand = "Set-Location '$project'; & '$python' -m celery -A app.tasks:celeryApp worker --pool=solo --loglevel=info"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $workerCommand

Write-Host "4/4 Starting the web app in a new window..."
$webCommand = "Set-Location '$project'; & '$python' run.py"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $webCommand

Write-Host "Waiting for the web app (the model takes up to a minute)..."
$ready = $false
for ($i = 0; $i -lt 40; $i++) {
    try {
        $reply = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 3
        if ($reply.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 3
    }
}

if ($ready) {
    Write-Host ""
    Write-Host "VisionGuard:   http://localhost:5000"
    Write-Host "Email inbox:   http://localhost:8025   (your login codes arrive here)"
    Write-Host "Neo4j browser: http://localhost:7474"
    Start-Process "http://localhost:5000"
} else {
    Write-Host "The web app did not answer. Look at the web window for the error."
}
```
Run it from your VS Code terminal (where you already set `RemoteSigned`):
```
.\dev.ps1
```
It opens two extra windows, one for the worker and one for the web app, and then opens the browser. Because it calls `.venv\Scripts\python.exe` directly, it doesn't depend on the virtual environment being active.

## To stop
Press Ctrl+C in the worker window and the web window, close them, then:
```
docker compose stop
```
`stop` keeps your data, and `down -v` would delete it.

## After you change code
- **Web app:** Ctrl+C in its window, then `python run.py` again.
- **Worker:** the worker does not reload by itself, so restart it too if you edit `app\tasks.py`, `app\graph.py` or `app\config.py`.

## One-off commands, not every day

| Command | When |
|---|---|
| `python -m src.index_knowledge` | After editing anything in `knowledge\` |
| `python -m src.graph_backfill` | After the database is reset, or after seeding demo data |

## Health check
`http://localhost:5000/health` should show `"status":"ok"` and `"redis":"ok"`. A Neo4j check isn't there yet, and I can add one on Day 7.

## Memory
About 0.7 GB for Neo4j, 2.5 GB for Ollama while Artist is answering, and about 1 GB for the web app with TensorFlow. That fits your 16 GB comfortably.

