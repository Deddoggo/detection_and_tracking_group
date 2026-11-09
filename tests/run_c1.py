"""Single CPU-only C1 command: python -B tests/run_c1.py (stdlib unittest)."""
from pathlib import Path
import importlib.metadata
import json
import platform
import sys
import unittest
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))
OUTPUT=ROOT/'.local/c1_validation'
OUTPUT.mkdir(parents=True,exist_ok=True)
import c1_cases


class Result(unittest.TextTestResult):
    records=[]
    def row(self,test,status,detail=None):
        self.records.append({'test_id':test.id(),'status':status,'detail':detail,
            'observations':c1_cases.EVIDENCE.get(test.id(),[]),
            'dependency_mode':getattr(test,'mode','native')})
    def addSuccess(self,test):
        super().addSuccess(test);self.row(test,'PASS')
    def addFailure(self,test,err):
        super().addFailure(test,err);self.row(test,'FAIL',self._exc_info_to_string(err,test))
    def addError(self,test,err):
        super().addError(test,err);self.row(test,'FAIL',self._exc_info_to_string(err,test))
    def addSkip(self,test,reason):
        super().addSkip(test,reason);self.row(test,'SKIP',reason)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(c1_cases)
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    versions={}
    for package in ('numpy','opencv-python','scikit-learn','pandas','openpyxl'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='not installed (distribution metadata)'
    payload={'command':f'"{sys.executable}" -B tests/run_c1.py','python':sys.version,
             'platform':platform.platform(),'packages':versions,'tests_run':result.testsRun,
             'records':result.records,'substitute_policy':'GraphDouble rows explicitly marked; native sklearn tests skip if absent'}
    payload['scope']='authoritative active pipeline only; no legacy/MixSTE tests'
    out=OUTPUT/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'.json')
    with out.open('x',encoding='utf8') as f:
        f.write(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print('Machine-readable test evidence: '+str(out))
    sys.exit(0 if result.wasSuccessful() else 1)
