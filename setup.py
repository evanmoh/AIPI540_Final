from setuptools import setup, Command
import subprocess

class RunAll(Command):
    description = 'Setup code - run all the codes'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Step 1: Data Prep code.
        print("Running scripts/data_prep.py ...")
        subprocess.check_call(['python', 'scripts/data_prep.py'])

        # Step 2: Feature Engineering code
        print("Running scripts/features.py ...")
        subprocess.check_call(['python', 'scripts/features.py'])

        # Step 3: SVD train code
        print("Running scripts/svd_train.py ...")
        subprocess.check_call(['python', 'scripts/features.py'])

        # Step 4: neural net
        print("Running scripts/ml.py ...")
        subprocess.check_call(['python', 'scripts/ml.py'])

        # Step 5: App
        print("Running app.py ...")
        subprocess.check_call(['python', 'app.py'])

setup(
    name='aipi540_final',
    version='0.1',
    cmdclass={
        'run': RunAll,
    },
)
