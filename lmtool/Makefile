UIFILE := $(wildcard *.ui)
PYFILE := $(UIFILE:.ui=.py)
QRCFILE := $(wildcard *.qrc)
RCFILE := $(QRCFILE:.qrc=_rc.py)

PYTHON = python
PYUIC = pyuic5
PYRCC = pyrcc5

UICOPTS = -x

.PHONY: all test clean

all: $(PYFILE) $(RCFILE)

test: $(PYFILE)
	$(PYTHON) $(PYFILE)

clean:
	-rm $(PYFILE) $(RCFILE)

%.py: %.ui
	$(PYUIC) $< $(UICOPTS) -o $@

%_rc.py: %.qrc
	$(PYRCC) $< -o $@
