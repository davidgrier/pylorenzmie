default: cminimizers

cminimizers: setup.py cminimizers.pyx
	python3 setup.py build_ext --inplace && rm -f cminimizers.c && rm -Rf build

clean:
	rm *.so *.html
