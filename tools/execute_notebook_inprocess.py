"""Execute cells in a fresh IPython process when local sockets are unavailable.

This executes real cell code and captures its real outputs; it is not a cached
output verifier. Standard Jupyter/nbclient remains the normal verification path.
"""
import argparse
import os
import sys
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


class InlineNotebookShell(InteractiveShell):
    def enable_gui(self, gui=None):
        if gui not in (None, 'inline'):
            raise RuntimeError('The socket-free executor supports inline figures only')
        self.active_eventloop = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook')
    parser.add_argument('output')
    args = parser.parse_args()
    source, output = Path(args.notebook).resolve(), Path(args.output).resolve()
    os.chdir(source.parent)
    sys.path.insert(0, str(source.parent))
    notebook = nbformat.read(source, as_version=4)
    shell = InlineNotebookShell.instance()
    # Configure inline figures without requiring GUI or socket services.
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline')
    from matplotlib_inline.backend_inline import configure_inline_support
    configure_inline_support(shell, 'inline')
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'{source.name}, code cell {count}: {captured.stdout}\n{captured.stderr}') from (result.error_before_exec or result.error_in_exec)
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for rich in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
    notebook.metadata['execution_validation'] = {'executor': 'fresh_process_ipython', 'code_cells': count}
    nbformat.validate(notebook)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)


if __name__ == '__main__':
    main()
