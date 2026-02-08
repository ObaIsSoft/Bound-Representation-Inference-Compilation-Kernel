
import asyncio
import sys
# Make sure we can import the test
import test_plan_workflow
from test_plan_workflow import test_plan_execution_stops_at_review, mock_registry_setup

async def run_test():
    print("Running verification test for /api/plan workflow...")
    
    # Manually run the test function with the fixture
    # Pytest fixtures are hard to invoke manually, so we'll just instantiate the context manager manually
    # or rely on running via pytest command.
    
    # Actually, running via pytest module is cleaner.
    import pytest
    retcode = pytest.main(["test_plan_workflow.py", "-v"])
    
    if retcode == 0:
        print("\n✅ Verification Successful: Workflow stopped at Review Gate as expected.")
    else:
        print("\n❌ Verification Failed.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_test())
