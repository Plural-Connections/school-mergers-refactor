import os, sys

sys.path.insert(0, os.path.abspath(".."))

project = "School Mergers"
copyright = "2025, Madison Landry"
author = "Madison Landry"
release = "v2025.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "enum_tools.autoenum",
]
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_default_flags = ["members", "undoc-members"]
autodoc_typehints = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_attr_annotations = True  # not working?
# napoleon_preprocess_types = True
# napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
