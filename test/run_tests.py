import sys
import pytest

if __name__ == "__main__":
    print("Running Neural Style Transfer tests...")

    # Add any command line arguments passed to this script
    args = sys.argv[1:] if len(sys.argv) > 1 else ["."]

    # Run the tests
    exit_code = pytest.main(args)

    # Exit with the pytest exit code
    sys.exit(exit_code)
