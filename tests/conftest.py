from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import run_pipeline


@pytest.fixture(scope="session")
def pipeline_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("aml-output")
    run_pipeline("data", output, logger=lambda _: None)
    return output
