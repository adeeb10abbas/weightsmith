"""Expose startup stalls without changing the original simulator entry point."""
import faulthandler
import runpy
import sys
faulthandler.enable()
faulthandler.dump_traceback_later(45,repeat=True)
entry=sys.argv.pop(1)
sys.argv[0]=entry
sys.path.insert(0,__import__('pathlib').Path(entry).parent.as_posix())
runpy.run_path(entry,run_name='__main__')
