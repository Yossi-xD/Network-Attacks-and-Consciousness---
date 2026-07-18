# Launches the Face Morph Studio demo app.
# Must run from the ASCII junction -- MediaPipe's native core can't start up
# when the process working directory contains non-ASCII characters (this
# folder's path has Hebrew characters); see README's Unicode/Windows note.
$junction = Join-Path $env:USERPROFILE "face_morph_project_run"
if (-not (Test-Path $junction)) {
    New-Item -ItemType Junction -Path $junction -Target $PSScriptRoot | Out-Null
}
$python = Join-Path $junction ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python was not found at $python. The junction may point to the wrong folder."
}

Push-Location $junction
try {
    & $python -m streamlit run "src\app.py"
}
finally {
    Pop-Location
}
