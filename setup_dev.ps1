# Simple Development Setup for Research Project
Write-Host "Setting up Research Project Development Tools" -ForegroundColor Green

# Install development dependencies
Write-Host "Installing dev tools (black, isort, flake8, pytest)..." -ForegroundColor Yellow
pip install -e ".[dev]"

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  Format code:    black src/ scripts/"
Write-Host "  Sort imports:   isort src/ scripts/"
Write-Host "  Check style:    flake8 src/ scripts/"
Write-Host "  Run tests:      pytest"
Write-Host "  Do all:         black src/ scripts/; isort src/ scripts/; flake8 src/ scripts/; pytest"
