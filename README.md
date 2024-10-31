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
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
   ```

6. Install Visual Studio Community 2022 (Windows only):
   - Download Visual Studio Community 2022 from the official Microsoft website: https://visualstudio.microsoft.com/vs/community/
   - Run the installer and select the following workloads:
     - Python development
     - Desktop development with C++
   - Complete the installation process

7. Install CUDA Toolkit 12.4 (Windows and Linux only):
   - For Windows:
     1. Download the CUDA Toolkit 12.4 installer from the NVIDIA website: https://developer.nvidia.com/cuda-12-4-0-download-archive
     2. Run the installer and follow the on-screen instructions.
     3. After installation, add CUDA to your system PATH:
        - Right-click on 'This PC' or 'My Computer' and select 'Properties'
        - Click on 'Advanced system settings'
        - Click on 'Environment Variables'
        - Under 'System variables', find and edit 'Path'
        - Add the following paths (adjust if you installed CUDA in a different location):
          ```
          C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
          C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp
          ```

   - For Linux:
     1. Download the CUDA Toolkit 12.4 runfile from the NVIDIA website: https://developer.nvidia.com/cuda-12-4-0-download-archive
     2. Make the runfile executable:
        ```
        chmod +x cuda_12.4.0_<version>_linux.run
        ```
     3. Run the installer:
        ```
        sudo sh cuda_12.4.0_<version>_linux.run
        ```
     4. Follow the on-screen instructions to complete the installation.
     5. Add CUDA to your PATH by adding these lines to your ~/.bashrc or ~/.zshrc file:
        ```
        export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
        ```
     6. Reload your shell configuration:
        ```
        source ~/.bashrc  # or source ~/.zshrc if you use zsh
        ```

   Note: Mac users can skip this step as macOS uses Metal Performance Shaders (MPS) for GPU acceleration, which doesn't require CUDA.

8. If you are using Windows, install PyTorch with CUDA support:
   ```
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

   Note: This step is only necessary for Windows users. Mac and Linux users can skip this step as the appropriate PyTorch version is already specified in the requirements.txt file.