#!/usr/bin/env bash

pip install -r requirements-dev.txt
pip install -e .
pre-commit install
npm i -g gitmoji-cli
gitmoji -i
