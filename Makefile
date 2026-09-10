# Two tiers, and the difference between them is what you have to download.
#
#   make figures    every figure and table in the paper, from what is in this repository.
#                   No prediction snapshot, no GPU, no cluster. About a minute.
#   make scoring    re-derive analysis/scoring/*_results.csv from the raw model predictions.
#                   Needs the 1.2 GB prediction snapshot; see analysis/README.md.
#                   0_ref is not in this target: rescoring NetMHCIIpan and MixMHC2pred needs
#                   those tools installed. Its output is tracked; see scoring/0_ref/README.md.
#   make datasets   rebuild the four dataset arms from the IEDB export. See pipeline/preprocess/README.md.
#   make supplementary  rebuild the machine-readable D01-D12 data set.
#   make verify     check the rebuilt datasets, the data set and the demo assets against what is
#                   published.
#   make demo-assets  rebuild the demo's HLA store from the research store.
#
# `figures` is the tier that matters for reading the paper: the scoring outputs are tracked, so the
# figure notebooks read them directly and nothing upstream has to be reproduced first.

PY ?= python
# --stdout, not --inplace: executing a notebook is how these targets produce their files, and
# writing the executed copy back would leave every run showing as a change to the notebook.
NB = jupyter nbconvert --to notebook --execute --stdout --ExecutePreprocessor.timeout=1800

FIGURE_NOTEBOOKS = \
	analysis/figures/tab2_pooled_benchmark.ipynb \
	analysis/figures/fig2_benchmark_panels.ipynb \
	analysis/figures/fig3_fig4_molecule_and_serotype.ipynb \
	analysis/figures/fig5_representation_size.ipynb \
	analysis/figures/figS1_dataset_overlap.ipynb \
	analysis/figures/paired_tests.ipynb

.PHONY: figures scoring datasets supplementary verify demo-assets clean-figures

figures:
	@for nb in $(FIGURE_NOTEBOOKS); do echo "-> $$nb"; (cd analysis/figures && $(NB) $$(basename $$nb) > /dev/null) || exit 1; done
	@echo "-> analysis/figures/figS2_umap.py"
	@cd analysis/figures && $(PY) figS2_umap.py
	@echo "-> analysis/figures/supp_ref_tables.py"
	@cd analysis/figures && $(PY) supp_ref_tables.py

scoring:
	@test -n "$$PREPIBIND_RAW_ROOT" || { echo "Set PREPIBIND_RAW_ROOT to the unpacked prediction snapshot."; exit 2; }
	@for d in analysis/scoring/*/anal_pred_*.ipynb; do echo "-> $$d"; (cd $$(dirname $$d) && $(NB) $$(basename $$d) > /dev/null) || exit 1; done
	@echo "-> analysis/scoring/8_strat"
	@$(PY) analysis/scoring/run.py 8_strat > /dev/null
	@echo "-> analysis/scoring/9_boot"
	@cd analysis/scoring/9_boot && $(PY) boot.py

# The arm notebooks write the *_beta.csv files themselves. build_beta.py reconstructs the same
# derivation for checking; running it with --write would replace four published files with
# content-equal but byte-different copies, so it stays out of this target. `make verify` runs it.
datasets:
	$(PY) pipeline/preprocess/run_all.py

supplementary:
	$(PY) supplementary_data/build_supplementary_data.py

demo-assets:
	$(PY) demo/build_demo_assets.py hla-store --write

verify:
	$(PY) pipeline/preprocess/verify_outputs.py
	$(PY) supplementary_data/build_supplementary_data.py --check
	$(PY) pipeline/preprocess/build_beta.py --check
	$(PY) pipeline/preprocess/apply_h2_correction.py --check
	$(PY) demo/build_demo_assets.py hla-store --check

clean-figures:
	rm -rf analysis/figures/data/extracted
