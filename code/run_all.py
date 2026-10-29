import eda
import holm
import prepare
import tests_ps1
import tests_ps2

for step in (prepare, eda, tests_ps1, tests_ps2, holm):
    print(f"\n=== {step.__name__} ===")
    step.main()