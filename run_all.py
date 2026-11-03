import runpy

for script in ["step2_load_check.py", "step3_core.py", "step4_space_quant_compare.py",
               "step5_whatif.py", "make_report.py"]:
    print(f"\n==================== {script} ====================")
    runpy.run_path(script, run_name="__main__")
