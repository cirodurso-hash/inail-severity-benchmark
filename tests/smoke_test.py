"""End-to-end smoke test on synthetic data (a few minutes):  python tests/smoke_test.py"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
subprocess.check_call([sys.executable, "tests/make_synthetic.py", "tests/synthetic"])
subprocess.check_call([sys.executable, "run.py", "all", "--config", "tests/smoke_config.yaml"])
expected = ["tables/Table1_records_removed.csv", "tables/Table2_multinomial_OR.csv", "tables/Table3_per_class.csv",
            "tables/Table4_performance.csv", "tables/Table5_profiles.csv", "supplementary/Supplementary_Tables.xlsx",
            "figures/Figure4_confusion_matrix.png", "figures/Figure5_forest_ordinal.png", "environment.json"]
missing = [p for p in expected if not os.path.exists(os.path.join("tests/out", p))]
print("SMOKE TEST", "FAILED, missing: %s" % missing if missing else "PASSED")
sys.exit(1 if missing else 0)
