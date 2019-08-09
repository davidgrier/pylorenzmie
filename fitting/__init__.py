import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pylorenzmie.fitting.cython.cminimizers import amoeba
except Exception as e:
    print(e)
    from pylorenzmie.fitting.minimizers import amoeba

all = ['mie_fit.py', 'minimizers.py', amoeba]
