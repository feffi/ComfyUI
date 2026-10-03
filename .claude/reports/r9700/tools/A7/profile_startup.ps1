<#
ComfyUI startup / time-to-first-image profiler for Windows. Read-only for ComfyUI and the venv:
it only runs Python with extra wrapper scripts and writes into -OutDir.

Usage (PowerShell, from anywhere):
  .\profile_startup.ps1 -ComfyDir C:\ComfyUI -Python C:\ComfyUI\.venv-rocm-100\Scripts\python.exe `
      -Runs 3 -Port 8199 -ComfyArgs @('--use-pytorch-cross-attention') [-Workflow C:\path\wf_api.json]

Cold vs warm:
  cold = first run right after a reboot (log in, wait ~2 min for login tasks, run with -Runs 1 -Label cold).
         Optional without reboot (admin): Sysinternals "RAMMap64.exe -Et" empties the standby list,
         which drops the file cache like a reboot does (Defender's scan cache is not reset by it).
  warm = run again immediately: -Runs 3 -Label warm. Report the median.
Use the same -ComfyArgs as production, but a free -Port, so the production instance is untouched.
Stop the production instance first if it runs on the same GPU, or the HIP init / probe timings mix with its load.
#>
param(
    [string]$ComfyDir = (Get-Location).Path,
    [string]$Python = "",
    [int]$Runs = 3,
    [int]$Port = 8199,
    [string[]]$ComfyArgs = @(),
    [string]$Workflow = "",
    [string]$Label = "warm",
    [string]$OutDir = (Join-Path $env:TEMP "comfy_startup_profile")
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Python) {
    $active = Join-Path $ComfyDir "config\rocm-stack.active"
    $Python = Join-Path $ComfyDir ".venv-rocm-100\Scripts\python.exe"
    if (Test-Path $active) { Write-Host "rocm-stack.active: $(Get-Content $active -Raw)" }
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$run = Join-Path $OutDir "$Label-$stamp"
New-Item -ItemType Directory -Force -Path $run | Out-Null

function Quote([string]$s) { if ($s -match '[\s"]') { '"' + ($s -replace '"', '\"') + '"' } else { $s } }
$comfyArgStr = (@("--port", "$Port") + $ComfyArgs | ForEach-Object { Quote $_ }) -join " "

# ---- environment report (read-only) ----
$env_txt = Join-Path $run "environment.txt"
& {
    "date: $(Get-Date -Format o)"
    $boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
    "last boot: $boot  (uptime $([int]((Get-Date) - $boot).TotalMinutes) min; first run after boot = cold)"
    "python: $Python"
    & $Python -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'hip', torch.version.hip); print('devices', [torch.cuda.get_device_properties(i).gcnArchName for i in range(torch.cuda.device_count())])"
    & $Python -c "import importlib.metadata as m; ds = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in m.distributions()}; [print(p, ds.get(p, 'not installed')) for p in ('torch', 'torchvision', 'torchaudio', 'rocm', 'rocm-sdk-core', 'comfy-kitchen', 'comfy-aimdo', 'triton-windows', 'comfyui-frontend-package', 'transformers', 'onnxruntime-directml', 'insightface', 'llama-cpp-python', 'uv', 'pip')]" 2>&1
    "PYTHONDONTWRITEBYTECODE=$env:PYTHONDONTWRITEBYTECODE  PYTHONPYCACHEPREFIX=$env:PYTHONPYCACHEPREFIX  UV_COMPILE_BYTECODE=$env:UV_COMPILE_BYTECODE"
    "HIP_VISIBLE_DEVICES=$env:HIP_VISIBLE_DEVICES  CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES  TRITON_CACHE_DIR=$env:TRITON_CACHE_DIR"
    try {
        $mp = Get-MpComputerStatus
        "Defender real-time protection: $($mp.RealTimeProtectionEnabled)  on-access: $($mp.OnAccessProtectionEnabled)"
        $pref = Get-MpPreference
        "Defender ExclusionPath: $($pref.ExclusionPath -join '; ')"
        "Defender ExclusionProcess: $($pref.ExclusionProcess -join '; ')"
    } catch { "Defender status not readable: $_" }
    foreach ($d in @($ComfyDir, (Split-Path $Python))) {
        $drive = (Get-Item $d).PSDrive.Name + ":"
        "fsutil devdrv query $drive :"
        try { fsutil devdrv query $drive 2>&1 } catch { "  (fsutil devdrv not available)" }
    }
    foreach ($ini in @("user\__manager\config.ini", "user\default\ComfyUI-Manager\config.ini")) {
        $p = Join-Path $ComfyDir $ini
        if (Test-Path $p) {
            "ComfyUI-Manager config: $p"
            Select-String -Path $p -Pattern '^(network_mode|use_uv|db_mode|file_logging)\s*=' | ForEach-Object { "  " + $_.Line }
        }
    }
    $tc = if ($env:TRITON_CACHE_DIR) { $env:TRITON_CACHE_DIR } else { Join-Path $env:USERPROFILE ".triton\cache" }
    if (Test-Path $tc) {
        $sz = (Get-ChildItem $tc -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum)
        "triton cache: $tc  files=$($sz.Count)  MB=$([int]($sz.Sum / 1MB))"
    } else { "triton cache: $tc (missing: first Triton use JIT-compiles everything)" }
} *>&1 | Tee-Object -FilePath $env_txt

# ---- 1. import time capture (raw stderr, so use Start-Process redirects) ----
Write-Host "`n== -X importtime (exits after node init via --quick-test-for-ci) =="
$imp = Join-Path $run "importtime.log"
$p = Start-Process -FilePath $Python -WorkingDirectory $ComfyDir -NoNewWindow -Wait -PassThru `
    -ArgumentList "-X importtime main.py --quick-test-for-ci $comfyArgStr" `
    -RedirectStandardError $imp -RedirectStandardOutput (Join-Path $run "importtime.stdout.log")
& $Python (Join-Path $Here "parse_importtime.py") $imp 30 | Tee-Object -FilePath (Join-Path $run "importtime_summary.txt")

# ---- 2. per-phase profile until "To see the GUI go to" ----
$jsons = @()
for ($i = 1; $i -le $Runs; $i++) {
    Write-Host "`n== startup run $i / $Runs =="
    $out = Join-Path $run "startup_$i"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $Python -WorkingDirectory $ComfyDir -NoNewWindow -Wait -PassThru `
        -ArgumentList "$(Quote (Join-Path $Here 'comfy_startup_profile.py')) --out $(Quote $out) -- $comfyArgStr" `
        -RedirectStandardError "$out.stderr.log" -RedirectStandardOutput "$out.stdout.log"
    Write-Host ("process wall incl. exit: {0:N0} ms" -f $sw.Elapsed.TotalMilliseconds)
    Get-Content "$out.txt" -TotalCount 8
    $jsons += "$out.json"
}
& $Python (Join-Path $Here "summarize_runs.py") @jsons | Tee-Object -FilePath (Join-Path $run "summary.txt")

# ---- 3. optional: launch -> first image ----
if ($Workflow) {
    Write-Host "`n== time to first image =="
    $t0 = [double]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()) / 1000.0
    $srv = Start-Process -FilePath $Python -WorkingDirectory $ComfyDir -NoNewWindow -PassThru `
        -ArgumentList "$(Quote (Join-Path $Here 'comfy_startup_profile.py')) --keep-running --out $(Quote (Join-Path $run 'startup_tti')) -- $comfyArgStr" `
        -RedirectStandardError (Join-Path $run "tti.stderr.log") -RedirectStandardOutput (Join-Path $run "tti.stdout.log")
    try {
        & $Python (Join-Path $Here "first_image.py") --workflow $Workflow --port $Port --t0 $t0 --out (Join-Path $run "first_image.json")
    } finally {
        Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "`nResults in $run"
