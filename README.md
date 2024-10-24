# Setup Instructions

1. Install Git LFS (Large File Storage):

   a. For Mac:
   
   <details>
   <summary>i. Install Homebrew:</summary>

   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
        
   After installing Homebrew, run the following commands.
   ```
   # Replace <username> with your actual username.
   
   echo >> /Users/<username>/.zprofile
   ```
        
   ```
   # Replace <username> with your actual username.
   
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/<username>/.zprofile
   ```
        
   ```
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```
   </details>

   <details>
   <summary>ii. Install Git LFS using Homebrew:</summary>

   ```
   brew install git-lfs
   ```
   </details>

   <details>
   <summary>iii. Set up Git LFS:</summary>

   ```
   git lfs install
   ```
   </details>

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

5. Download NLTK stopwords and punkt:
   ```
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
   ```