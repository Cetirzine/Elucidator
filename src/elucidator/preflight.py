import importlib
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from elucidator.settings import Settings

app = typer.Typer(add_completion=False, help="Validate the local research environment.")
console = Console()

runtime_cache = Path("data/cache")
(runtime_cache / "matplotlib").mkdir(parents=True, exist_ok=True)
(runtime_cache / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(runtime_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(runtime_cache / "xdg"))

IMPORTS = {
    "AKShare": "akshare",
    "BaoStock": "baostock",
    "CatBoost": "catboost",
    "DuckDB": "duckdb",
    "LightGBM": "lightgbm",
    "MLflow": "mlflow",
    "OpenAI client": "openai",
    "Optuna": "optuna",
    "Pandas": "pandas",
    "Prophet": "prophet",
    "PyArrow": "pyarrow",
    "Qlib": "qlib",
    "scikit-learn": "sklearn",
    "SHAP": "shap",
    "Torch": "torch",
    "Tushare": "tushare",
    "XGBoost": "xgboost",
}


@app.command()
def main(
    check_api: bool = typer.Option(False, help="Also authenticate against the LLM API."),
) -> None:
    table = Table(title="Elucidator dependency preflight")
    table.add_column("Component")
    table.add_column("Status")
    failures: list[str] = []

    if sys.version_info[:2] != (3, 12):
        failures.append(f"Python {sys.version.split()[0]} (requires 3.12.x)")
        table.add_row("Python", f"FAIL: {sys.version.split()[0]}")
    else:
        table.add_row("Python", sys.version.split()[0])

    for label, module_name in IMPORTS.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "imported")
            table.add_row(label, str(version))
        except Exception as exc:  # noqa: BLE001 - preflight must report binary import failures too
            failures.append(f"{label}: {exc}")
            table.add_row(label, f"FAIL: {exc}")

    if check_api:
        settings = Settings()
        if settings.llm_api_key is None:
            failures.append("LLM API: missing LLM_API_KEY")
            table.add_row("LLM API", "FAIL: missing key")
        else:
            from openai import OpenAI

            try:
                client = OpenAI(
                    api_key=settings.llm_api_key.get_secret_value(),
                    base_url=settings.llm_base_url,
                    timeout=settings.llm_timeout_seconds,
                )
                model_ids = {model.id for model in client.models.list().data}
                status = (
                    "available" if settings.llm_model in model_ids else "model alias not listed"
                )
                table.add_row("LLM API", status)
                if settings.llm_model not in model_ids:
                    failures.append(f"LLM API: {settings.llm_model!r} not returned by /models")
            except Exception as exc:  # noqa: BLE001 - display API/network/auth failures uniformly
                failures.append(f"LLM API: {exc}")
                table.add_row("LLM API", f"FAIL: {exc}")

    console.print(table)
    if failures:
        console.print("\n[red]Preflight failed:[/red]")
        for failure in failures:
            console.print(f"- {failure}")
        raise typer.Exit(code=1)
    console.print("\n[green]All requested checks passed.[/green]")


if __name__ == "__main__":
    app()
