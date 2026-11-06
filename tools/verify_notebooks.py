from pathlib import Path
import os,sys,json,time,subprocess
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYDEVD_DISABLE_FILE_VALIDATION']='1'
for key,sub in [('IPYTHONDIR','ipython'),('JUPYTER_CONFIG_DIR','jupyter_config'),('JUPYTER_DATA_DIR','jupyter_data'),('JUPYTER_RUNTIME_DIR','jupyter_runtime'),('MPLCONFIGDIR','matplotlib')]:
    p=R/'outputs/_runtime'/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
import nbformat
from nbclient import NotebookClient
from traitlets.config import Config
from briefing_utils import verify_assets,check_stage_evidence
def main():
    count=verify_assets()
    evidence=check_stage_evidence()
    files=sorted(R.glob('*.ipynb'))
    assert [p.name for p in files]==['01_act_imitation_learning.ipynb','02_worldgym_policy_evaluation.ipynb','03_latent_pickcube_policy_evaluation.ipynb','04_latent_tube_policy_evaluation.ipynb','05_dyna_style_closed_loop.ipynb']
    result=[]
    executor=os.environ.get('AUTOBIO_NOTEBOOK_EXECUTOR','nbclient')
    assert executor in ('nbclient','inprocess')
    kernel_config=Config()
    if os.environ.get('AUTOBIO_KERNEL_TRANSPORT')=='ipc':
        kernel_config.KernelManager.transport='ipc'
        kernel_config.KernelManager.ip=f'/tmp/autobio-notebooks-{os.getpid()}'
    for path in files:
        nb=nbformat.read(path,as_version=4);start=time.perf_counter()
        if executor=='inprocess':
            subprocess.run([sys.executable,str(R/'tools/execute_notebook_inprocess.py'),str(path),str(R/'outputs'/path.name)],check=True,cwd=R,timeout=180)
            nb=nbformat.read(R/'outputs'/path.name,as_version=4)
        else:
            NotebookClient(nb,timeout=180,kernel_name='python3',config=kernel_config,allow_errors=False,resources={'metadata':{'path':str(R)}}).execute()
        nbformat.validate(nb)
        cells=[c for c in nb.cells if c.cell_type=='code']
        assert all(c.execution_count for c in cells)
        assert not any(o.output_type=='error' for c in cells for o in c.outputs)
        (R/'outputs'/path.name).write_text(nbformat.writes(nb)+chr(10),encoding='utf-8',newline=chr(10))
        result.append(dict(notebook=path.name,status='PASS',executor=executor,code_cells=len(cells),seconds=time.perf_counter()-start))
        print(path.name+': PASS',flush=True)
    (R/'outputs/NOTEBOOK_CHECK.json').write_text(json.dumps(dict(status='PASS',assets=count,evidence=evidence,notebooks=result,new_inference=False),indent=2),encoding='utf-8')
if __name__=='__main__':main()
