# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.append ('..')
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ZLEL'
copyright = '2025, Alvaro Beltran De Nanclares (abeltrandenanc002@ikasle.ehu.eus), Eneko Aranburu (earanburu006@gmail.com)'
author = 'Alvaro Beltran De Nanclares (abeltrandenanc002@ikasle.ehu.eus), Eneko Aranburu (earanburu006@gmail.com)'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
 'sphinx.ext.doctest',
 'sphinx.ext.intersphinx',
 'sphinx.ext.todo',
 'sphinx.ext.coverage',
 'sphinx.ext.mathjax',
 'sphinx.ext.ifconfig',
 'sphinx.ext.viewcode',
 'sphinx.ext.githubpages', ]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
