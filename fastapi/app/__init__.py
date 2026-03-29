"""Pulsecast FastAPI application package."""

import warnings

# Runs before app.main (and any router) imports LangChain. LangChain still touches
# Pydantic v1 on import; Python 3.14+ emits UserWarning until upstream removes it.
warnings.filterwarnings(
    "ignore",
    message=r".*Pydantic V1.*Python 3\.1[4-9].*",
    category=UserWarning,
    module=r"langchain_core\._api\.deprecation",
)
