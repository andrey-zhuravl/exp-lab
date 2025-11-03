import typer

from exp_lab.pipeline import report_experiment, run_experiment

app = typer.Typer(help="exp-lab CLI")


@app.command()
def run(m: str = typer.Option(..., "--manifest", "-m", help="Path to experiment manifest YAML")):
    """Run full A1 pipeline."""
    run_experiment(m)


@app.command()
def report(m: str = typer.Option(..., "--manifest", "-m", help="Path to experiment manifest YAML")):
    """Generate/print report for last run of this manifest."""
    report_experiment(m)


if __name__ == "__main__":
    app()
