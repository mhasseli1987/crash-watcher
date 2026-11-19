
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
& '.\.venv\Scripts\pythonw.exe' 'desktop_app.py'
