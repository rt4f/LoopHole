# Notebooks

Exploratory analysis only. Anything that other work depends on belongs in a
package or an experiment script.

`*.ipynb` files are git-ignored (see the root `.gitignore`) to keep outputs and
diffs out of history. To share a notebook, commit it as a
[jupytext](https://jupytext.readthedocs.io/) percent-format `.py` file:

```bash
pip install jupytext
jupytext --to py:percent analysis.ipynb
```

Name notebooks `YYYY-MM-DD-topic.py`.
