# A6 system check (read-only). Run in PowerShell from the ComfyUI folder:
#   powershell -ExecutionPolicy Bypass -File a6_system_check.ps1 [-Python .\.venv-rocm-100\Scripts\python.exe] [-Log comfyui.log]
# Reports TDR registry values, GPU driver version / PCI location per adapter, display power timeout,
# then runs a6_stability_check.py. Changes nothing.
param(
    [string]$Python = ".\.venv-rocm-100\Scripts\python.exe",
    [string]$Log = ""
)

Write-Host "== TDR (HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers); absent = Windows default"
$gd = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -ErrorAction SilentlyContinue
foreach ($k in "TdrLevel", "TdrDelay", "TdrDdiDelay", "TdrLimitCount", "TdrLimitTime", "TdrDebugMode") {
    $v = $gd.$k
    if ($null -eq $v) { $v = "(not set)" }
    "{0,-14} {1}" -f $k, $v
}
"Defaults: TdrLevel 3, TdrDelay 2 s, TdrDdiDelay 5 s, TdrLimitCount 5 per TdrLimitTime 60 s."

Write-Host "`n== GPUs (driver 32.0.31041.1004 = Adrenalin 26.8.1; 32.0.32015.2008 = PRO 26.9.2 with the idle page-out fix)"
Get-CimInstance Win32_VideoController | ForEach-Object {
    $loc = (Get-PnpDeviceProperty -InstanceId $_.PNPDeviceID -KeyName DEVPKEY_Device_LocationInfo -ErrorAction SilentlyContinue).Data
    "{0} | driver {1} ({2}) | {3} | displays: {4}x{5}" -f $_.Name, $_.DriverVersion, $_.DriverDate, $loc, $_.CurrentHorizontalResolution, $_.CurrentVerticalResolution
}

Write-Host "`n== Display / sleep timeouts on AC (a headless card entering D3 triggers the 26.5.1-26.8.1 VRAM page-out)"
powercfg /query SCHEME_CURRENT SUB_VIDEO VIDEOIDLE | Select-String "Current AC Power Setting Index"
powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String "Current AC Power Setting Index"

Write-Host "`n== Recent display-driver resets (Event ID 4101, last 30 days)"
Get-WinEvent -FilterHashtable @{LogName = "System"; Id = 4101; StartTime = (Get-Date).AddDays(-30)} -ErrorAction SilentlyContinue |
    Select-Object -First 10 TimeCreated, ProviderName, Message | Format-Table -AutoSize -Wrap
Write-Host "Bugchecks 0x116/0x117 (WER, last 30 days):"
Get-WinEvent -FilterHashtable @{LogName = "System"; Id = 1001; StartTime = (Get-Date).AddDays(-30)} -ErrorAction SilentlyContinue |
    Where-Object { $_.Message -match "0x00000116|0x00000117|0x0000007e" } | Select-Object -First 10 TimeCreated, Message | Format-List

Write-Host "`n== RAM"
$cs = Get-CimInstance Win32_ComputerSystem
"Installed RAM: {0:N1} GiB" -f ($cs.TotalPhysicalMemory / 1GB)

$args2 = @("$PSScriptRoot\a6_stability_check.py", "--comfy", ".")
if ($Log -ne "") { $args2 += @("--log", $Log) }
& $Python @args2
