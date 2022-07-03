help:
	@echo "available commands"
	@echo " - dev               : creates the development environment"
	@echo " - clean             : clean temporary folders and files"
	@echo " - lint              : checks code style and type checks"
	@echo " - clean-all         : removes environment, volumes, containers and images"

dev:
	pip install --upgrade pip && \
		pip install -r requirements-dev.txt && \
		pip install -e . && \
		pre-commit install && \
		((command -v gitmoji >/dev/null && gitmoji -i) || echo Please install gitmoji-cli)

clean:
	rm -rf `find . -type d -name .pytest_cache`
	rm -rf `find . -type d -name .mypy_cache`
	rm -rf `find . -type d -name __pycache__`
	rm -rf `find . -type d -name .ipynb_checkpoints`
	rm -f .coverage

lint: clean
	flake8; mypy; pydocstyle; black vxr

clean-all: clean
	rm -rf app.egg-info
	rm -rf env
