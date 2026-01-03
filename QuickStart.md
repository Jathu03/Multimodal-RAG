```markdown
## ⚡ Quick Start (No Coding Required)

Want to run this application without installing Python or cloning the entire source code?

### 📋 Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

---

### 🚀 Setup Instructions

#### 1. Download the Runner File
Open your terminal/command prompt and run:
```bash
curl -o docker-compose.yml [https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-REPO/main/docker-compose.prod.yml](https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-REPO/main/docker-compose.prod.yml)

```

#### 2. Set your API Key

Replace `your-key-here` with your actual Google API key.

**Linux / macOS:**

```bash
export GOOGLE_API_KEY=your-key-here

```

**Windows (PowerShell):**

```powershell
$env:GOOGLE_API_KEY="your-key-here"

```

#### 3. Run the App

Execute the following command to pull the images and start the services:

```bash
docker-compose up -d

```

---

### 🌐 Access the Application

Once the setup is complete, you can access the services at:

* **Frontend:** [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)
* **Backend API Docs:** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)

```

**Would you like me to help you create a `docker-compose.yml` file that matches this setup?**

```
