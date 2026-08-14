# Nap .env vao phien PowerShell hien tai.
#
# Repo nay KHONG co python-dotenv: `RealModel.from_env` doc thang
# os.environ (arena/model.py:839), nen file .env phai duoc nap vao tien
# trinh truoc khi chay run_practice.py.
#
#     .\load_env.ps1
#     python scripts/run_practice.py --model real --prompt-addendum
#
# Bien chi song trong cua so PowerShell hien tai. Mo cua so moi thi chay lai.

param([string]$Path = ".env")

if (-not (Test-Path $Path)) {
    Write-Error "Khong thay $Path. Tao no tu .env.example roi dien ba bien ARENA_*."
    exit 1
}

$loaded = @()
foreach ($line in Get-Content $Path -Encoding UTF8) {
    $trimmed = $line.Trim()
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
    $split = $trimmed.IndexOf("=")
    if ($split -lt 1) { continue }
    $name = $trimmed.Substring(0, $split).Trim()
    $value = $trimmed.Substring($split + 1).Trim().Trim('"').Trim("'")
    # Bo qua gia tri con rong hoac con la placeholder chua thay (vi du
    # "https://[host]/v1" hay "sk-..."): nap placeholder vao con te hon
    # de trong, vi loi se hien ra o tan tang HTTP.
    if ($value -eq "") { continue }
    if ($value.Contains([char]0x3C) -or $value.Contains([char]0x3E)) { continue }
    if ($value -eq "sk-...") { continue }
    Set-Item -Path "env:$name" -Value $value
    $loaded += $name
}

$required = @("ARENA_API_KEY", "ARENA_BASE_URL", "ARENA_MODEL")
$missing = $required | Where-Object { -not (Test-Path "env:$_") }

if ($missing.Count -gt 0) {
    Write-Error ("Chua dien gia tri cho: " + ($missing -join ", ") + " -- mo .env va dien not.")
    exit 1
}

# In ten bien, KHONG in gia tri cua key: no khong nen nam trong scrollback.
Write-Host ("Da nap: " + ($loaded -join ", "))
Write-Host ("  ARENA_BASE_URL = " + $env:ARENA_BASE_URL)
Write-Host ("  ARENA_MODEL    = " + $env:ARENA_MODEL)
Write-Host ("  ARENA_API_KEY  = da dat, " + $env:ARENA_API_KEY.Length + " ky tu")
