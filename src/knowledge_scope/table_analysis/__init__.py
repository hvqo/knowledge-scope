"""Deterministic PDF table analysis (M2).

The package turns a table Evidence item into a normalized, numeric
:class:`~knowledge_scope.table_analysis.models.StructuredTable` and exposes a
small set of fixed operations over it.  It never evaluates model-provided code,
SQL or expressions; every result is traceable to its Evidence lineage.
"""

from knowledge_scope.table_analysis.models import (
    TABLE_ANALYSIS_SCHEMA_VERSION,
    TableAnalysisRequest,
    TableAnalysisResult,
)
from knowledge_scope.table_analysis.service import (
    TableAnalysisService,
    analyze_table_evidence,
)

__all__ = [
    "TABLE_ANALYSIS_SCHEMA_VERSION",
    "TableAnalysisRequest",
    "TableAnalysisResult",
    "TableAnalysisService",
    "analyze_table_evidence",
]
