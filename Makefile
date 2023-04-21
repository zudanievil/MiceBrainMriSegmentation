#! /usr/bin/env make
# lists some common tasks. serves mostly as a memo
PROJ_NAME=BrainSegmentation

default_test:
	pytest -c pyproject.toml
visual_tests:
	pytest tests/* -m 'visual'
build_package:
	python -m build
clean_after_build:
	rm -rv $(PROJ_NAME).egg-info dist
install_built_package:
	pip install ./dist/$(PROJ_NAME)*.whl
install_editable:
	pip install -e .
reflow_the_code:
	black --config pyproject.toml .
uninstall:
	pip uninstall $(PROJ_NAME) -y


.PHONY: default_test visual_tests build_package clean_after_build install_built_package install_editable uninstall