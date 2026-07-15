import pandas as pd
import pytest

from tumor_subtyper.embedding import get_embedding_scvi


def test_scvi_dependency_error_is_actionable(monkeypatch):
    import tumor_subtyper.embedding as module

    def missing():
        raise ImportError("scVI support is not installed. Install the scvi extra.")

    monkeypatch.setattr(module, "_scvi_imports", missing)
    data = pd.DataFrame([[1, 2]], index=["sample"], columns=["gene1", "gene2"])
    with pytest.raises(ImportError, match="scvi extra"):
        get_embedding_scvi(data, "missing-model")
