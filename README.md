<p align = "center">
  <img src = "images/sensei_icon.png" alt = "SENSEI Logo" width = "200" style = "border-radius: 24px;"/>
</p>

<p align = "center">
<pre align="center">
███████╗███████╗███╗   ██╗███████╗███████╗██╗
██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝██║
███████╗█████╗  ██╔██╗ ██║███████╗█████╗  ██║
╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  ██║
███████║███████╗██║ ╚████║███████║███████╗██║
╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝
</pre>
</p>

<p align = "center">
  <b>activity Sequence ENcoder through maSkEd activIty structure modeling</b><br>
  <i>A Next-Generation Process Mining Language Model</i>
</p>

<p align = "center">
  <a href = "https://www.python.org/"><img src = "https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white" alt = "Python"/></a>
  <a href = "https://pytorch.org/"><img src = "https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt = "PyTorch"/></a>
  <a href = "https://huggingface.co/transformers/"><img src = "https://img.shields.io/badge/%F0%9F%A4%97_Transformers-FFD21E.svg?style=for-the-badge&logoColor=black" alt = "Transformers"/></a>
</p>

---

## Setup Instructions

This project requires Python 3.8+ and uses a virtual environment named `venv_SENSEI` to isolate dependencies.

### 1. Clone the Repository
```bash
git clone https://github.com/EmanuelePiemontese/SENSEI.git
cd "SENSEI"
```

### 2. Create the Virtual Environment

Initialize the virtual environment from the root directory of the project:

**On Windows:**

```bash
python -m venv venv_SENSEI
```

**On macOS / Linux:**

```bash
python3 -m venv venv_SENSEI
```

### 3. Activate the Virtual Environment

Before running scripts or installing packages, you must activate the environment:

**On Windows (Command Prompt):**

```cmd
venv_SENSEI\Scripts\activate.bat
```

**On Windows (PowerShell):**

```powershell
venv_SENSEI\Scripts\Activate.ps1
```

**On macOS / Linux:**

```bash
source venv_SENSEI/bin/activate
```

Once activated, your terminal prompt will show `(venv_SENSEI)`.

### 4. Install Dependencies

Install all the required external libraries using the provided `requirements.txt` file:

```bash
pip install -r "requirements.txt"
```

### 5. Deactivation

When you are done working on the project, you can exit the virtual environment by running:

```bash
deactivate
```