#!/usr/bin/env python3
"""Run all reproducibility scripts and collect eight manuscript figures under figures/."""
from pathlib import Path
import shutil
import subprocess
import sys
P=Path(__file__).resolve().parent
C=P/'code';FIG=P/'figures';FIG.mkdir(exist_ok=True)
for script in ('verify_new_theorems.py','closure_and_mutation.py','independent_controls.py','new_checks.py','fast_dim4_check.py','fast_generate_dim4.py'):
 print('Running',script,flush=True)
 subprocess.run([sys.executable,str(C/script)],check=True,cwd=C)
mapnames={
 'fig1_unimodal_same_initial.png':'fig1_unimodal_same_initial.png',
 'fig2_fitness_gap.png':'fig2_fitness_gap.png',
 'fig4_false_certification.png':'fig3_false_certification.png',
 'fig3_generic_boundary_trap.png':'fig4_generic_boundary_trap.png',
 'fig5_quantified_loss.png':'fig5_quantified_loss.png',
 'fig6_false_invariant.png':'fig6_false_invariant.png',
 'fig7_dim4_physical_time_fast.png':'fig7_dim4_physical_time.png',
 'fig7_mutation_restores_motion.png':'fig8_mutation_restores_motion.png',
}
for old,new in mapnames.items():
 source=C/'figures'/old
 if not source.exists(): raise FileNotFoundError(source)
 shutil.copy2(source,FIG/new)
 print('Collected',new)
R=P/'results';R.mkdir(exist_ok=True)
for old,new in [('new_checks_results.json','fitness_loss_check.json')]:
 if (C/old).exists(): shutil.copy2(C/old,R/new)
# For standalone scripts the new_checks.py writes its JSON in code/results.json.
if (C/'results.json').exists():shutil.copy2(C/'results.json',R/'fitness_loss_check.json')
if (C/'dim4_fast_results.json').exists():shutil.copy2(C/'dim4_fast_results.json',R/'dim4_fast_results.json')
print('Success: 8 figures present in',FIG)
