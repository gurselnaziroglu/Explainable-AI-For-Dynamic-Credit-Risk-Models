import sys

REQUIRED_PYTHON = "python3.10"


def main():
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor
    if REQUIRED_PYTHON == "python3":
        minimum_major = 3
        minimum_minor = 0
    elif REQUIRED_PYTHON == "python3.10":
        minimum_major = 3
        minimum_minor = 10
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(REQUIRED_PYTHON))

    if system_major < minimum_major or system_minor < minimum_minor:
        raise TypeError(
            f"This project requires Python {minimum_major}.{minimum_minor}. Found: Python {system_major}.{system_minor}"
        )
    else:
        print(">>> Development environment passes all tests!")


if __name__ == "__main__":
    main()
