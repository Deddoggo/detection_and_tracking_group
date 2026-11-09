"""C2 runner: use the paper-research interpreter explicitly; never installs dependencies."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))
from c2_pipeline import provenance
import test_c2


class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.records=[]

    def addSuccess(self,test):
        super().addSuccess(test)
        self.records.append({'test':test.id(),'status':'PASS'})

    def addError(self,test,error):
        super().addError(test,error)
        self.records.append({'test':test.id(),'status':'ERROR','detail':self._exc_info_to_string(error,test)})

    def addFailure(self,test,error):
        super().addFailure(test,error)
        self.records.append({'test':test.id(),'status':'FAIL','detail':self._exc_info_to_string(error,test)})

    def addSkip(self,test,reason):
        super().addSkip(test,reason)
        self.records.append({'test':test.id(),'status':'SKIP','detail':reason})


if __name__=='__main__':
    if Path(sys.prefix).name != 'paper-research':
        raise SystemExit('Use the existing paper-research interpreter; no fallback environment is selected')
    # Keep disposable test writes inside the ignored workspace, including on Windows.
    import tempfile
    temporary = ROOT/'.local/c2/test_temp'
    temporary.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir = str(temporary)
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_c2)
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    payload={'provenance':provenance(),'tests_run':result.testsRun,'records':result.records,
             'success':result.wasSuccessful(),'scope':'native CPU contracts and sklearn; explicitly synthetic poses; local XML where present'}
    target=ROOT/'.local/c2'/('tests_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
    with target.open('x',encoding='utf8') as f:
        json.dump(payload,f,indent=2,allow_nan=False)
    print('Machine-readable evidence: '+str(target))
    sys.exit(0 if result.wasSuccessful() else 1)
