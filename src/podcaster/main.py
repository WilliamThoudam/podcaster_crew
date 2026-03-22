#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from podcaster.crew import Podcaster

import os

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def _crew_inputs() -> dict:
    """Shared kickoff inputs; host names match .env (MALE_HOST / FEMALE_HOST)."""
    return {
        "topic": os.getenv("TOPIC"),
        "current_month": str(datetime.now().month),
        "current_year": str(datetime.now().year),
        "male_host": os.getenv("MALE_HOST", "Jone"),
        "female_host": os.getenv("FEMALE_HOST", "Jane"),
    }


# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = _crew_inputs()

    try:
        Podcaster().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = _crew_inputs()
    try:
        Podcaster().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Podcaster().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = _crew_inputs()

    try:
        Podcaster().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
