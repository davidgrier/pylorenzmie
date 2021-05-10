#PYTHON ?= python3

uname_s := $(shell uname -s)

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

COVERAGE = coverage-3.7
COVERAGE_TEST = $(COVERAGE) run -m unittest discover
COVERAGE_REPORT = $(COVERAGE) report -m

ifeq ($(uname_s),Darwin)
	export DYLD_FALLBACK_LIBRARY_PATH := $(CUDA_LIB)
endif

all: test

test:
	$(COVERAGE_TEST)
	$(COVERAGE_REPORT)
