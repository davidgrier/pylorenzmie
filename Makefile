#PYTHON ?= python3

uname_s := $(shell uname -s)

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

COVERAGE = coverage
COVERAGE_TEST = $(COVERAGE) run -m unittest discover
COVERAGE_REPORT = $(COVERAGE) report -m --include="$(ROOT_DIR)/*"

ifeq ($(uname_s),Darwin)
	export DYLD_FALLBACK_LIBRARY_PATH := $(CUDA_LIB)
endif

all: test

junk:
	echo $(ROOT_DIR)

test:
	$(COVERAGE_TEST)
	$(COVERAGE_REPORT)
