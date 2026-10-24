"""
oppti_immune — Pan-cancer proteogenomic outlier analysis of immune evasion proteins.

Modules
-------
preprocessing  : Normalization, QC, and filtering
concordance    : RNA–protein concordance classification
pies           : Protein-level Immune Evasion Score computation + validation
utils          : Shared utilities (config loading, logging, I/O)
"""

__version__ = "1.0.0"
__author__  = "Abdulkadir Elmas"

from .utils import load_config, get_logger  # noqa: F401
