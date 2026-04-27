# Start DocMind Frontend & Backend
Write-Host "=========================================="
Write-Host "    DocMind Startup Script (Decoupled)   "
Write-Host "=========================================="
Write-Host ""

$baseDir = "e:\BaiduNetdiskDownload\LLM\docmind"
Set-Location $baseDir

# Kill existing processes if any (optional but recommended for dev)
Write-Host "Cleaning up old processes..."
Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*docmind*" } | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "[1] Starting FastAPI Backend (Port 8000)..."
# Using cmd /k to keep window open if crash, and activate venv first
Start-Process -FilePath "cmd.exe" -ArgumentList "/k `"venv\Scripts\activate.bat & python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload`"" -WorkingDirectory $baseDir

Start-Sleep -Seconds 3

Write-Host "[2] Starting Streamlit Frontend (Port 8501)..."
Start-Process -FilePath "cmd.exe" -ArgumentList "/k `"venv\Scripts\activate.bat & streamlit run app.py --server.port 8501`"" -WorkingDirectory $baseDir

Write-Host ""
Write-Host "All services started!"
Write-Host "FastAPI Backend: http://localhost:8000/docs"
Write-Host "Streamlit Frontend: http://localhost:8501"
Write-Host ""
Write-Host "NOTE: Check the popup windows if something is not working."
