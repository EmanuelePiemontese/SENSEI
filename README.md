                            ███████╗███████╗███╗   ██╗███████╗███████╗██╗
                            ██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝██║
                            ███████╗█████╗  ██╔██╗ ██║███████╗█████╗  ██║
                            ╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝  ██║
                            ███████║███████╗██║ ╚████║███████║███████╗██║
                            ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝

## Setup Instructions

This project requires Python 3.8+ and uses a virtual environment named `venv_SENSEI` to isolate dependencies.

### 1. Clone the Repository
```bash
git clone "[https://github.com/EmanuelePiemontese/SENSEI.git](https://github.com/EmanuelePiemontese/SENSEI.git)"
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