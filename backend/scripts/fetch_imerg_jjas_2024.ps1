$ErrorActionPreference = "Stop"
$backend = Split-Path -Parent $PSScriptRoot
Set-Location $backend

Write-Host "Earthdata token input is hidden and is not written to disk."
$secure = Read-Host "Paste a valid EARTHDATA_TOKEN" -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try {
    $env:EARTHDATA_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    & .venv\Scripts\python.exe -m monsoonpp.tools.fetch_imerg `
        --start 2024-06-01 --end 2024-09-30 --out data\raw\imerg --threads 8
    if ($LASTEXITCODE -ne 0) {
        throw "IMERG download failed with exit code $LASTEXITCODE"
    }
}
finally {
    Remove-Item Env:\EARTHDATA_TOKEN -ErrorAction SilentlyContinue
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    $secure.Dispose()
}

Write-Host "JJAS IMERG download and HDF5 signature verification complete."
Read-Host "Press Enter to close"
