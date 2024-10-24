# Setup Instructions

1. Install Git LFS (Large File Storage):
   ```
   git lfs install
   ```

2. Create a Conda environment:
   ```
   conda create --prefix=venv python=3.11 -y
   ```

3. Activate the Conda environment:
   ```
   conda activate ./venv
   ```

4. Install required packages:
   ```
   python -m pip install --force-reinstall -r requirements.txt
   ```