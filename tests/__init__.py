import logging

# The code under test logs warnings on purpose; keep them out of the test report.
logging.disable(logging.CRITICAL)
