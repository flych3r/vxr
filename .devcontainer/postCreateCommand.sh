#!/usr/bin/env bash

pip install -r requirements-dev.txt
pip install -e .
pre-commit install
npm install --location=global gitmoji-cli
npm install --location=global git-commit-msg-linter
gitmoji -i
