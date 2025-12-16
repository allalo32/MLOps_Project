import evidently
print(f"Evidently version: {evidently.__version__}")
try:
    from evidently.report import Report
    print("Report import successful")
except ImportError as e:
    print(f"Report import failed: {e}")
