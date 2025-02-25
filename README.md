# 📝 AI Resume Screening System  


An AI-powered **resume parsing and screening system** that automates **candidate analysis** using **NLP, FAISS, and LangChain**. This application allows recruiters to **upload resumes, extract key details, and ask AI-powered questions** about candidates' qualifications.

---

## 🚀 Features  
✅ **AI-Powered Resume Parsing** – Extracts **Name, Experience, Skills, Projects, Education, Certifications**.  
✅ **AI Q&A System** – Ask **job-specific questions** about a resume and get precise AI-generated answers.  
✅ **Candidate Matching** – Compares skills against job descriptions for role suitability.  
✅ **Interactive Web Interface** – Built with **Streamlit** for an intuitive recruiter-friendly experience.  
✅ **Real-Time Processing** – Supports **PDF & Text resumes**, providing instant results.  

---

## 🏗️ Tech Stack  
🔹 **Python** (Core Programming)  
🔹 **Streamlit** (Web App)  
🔹 **FAISS** (Vector Search)  
🔹 **LangChain** (AI Q&A)  
🔹 **PyPDF2** (Resume Parsing)  
🔹 **Hugging Face Embeddings** (AI Processing)  
🔹 **Groq API** (LLM Integration)  

---

## 📂 Project Structure  
AI-Resume-Screening/ │── app.py # Main Streamlit application │── qa_system.py # AI-powered Q&A system │── utils.py # Groq API interaction functions │── test.py # CUDA and PyTorch testing │── requirements.txt # Project dependencies │── README.md # Documentation

yaml
Copy
Edit

---

## 🔧 Installation & Setup  

### **1️⃣ Clone the Repository**  
git clone https://github.com/rishuSingh404/AI-Resume-Screening.git cd AI-Resume-Screening

mathematica
Copy
Edit

### **2️⃣ Set Up Virtual Environment**  
python -m venv venv source venv/bin/activate # Mac/Linux venv\Scripts\activate # Windows

markdown
Copy
Edit

### **3️⃣ Install Dependencies**  
pip install -r requirements.txt

markdown
Copy
Edit

### **4️⃣ Set Up API Keys**  
Create a `.env` file in the project root and add:  

yaml
Copy
Edit

---

## 🚀 Running the Application  

### **Run the Streamlit Web App**  
streamlit run app.py

yaml
Copy
Edit
🔹 This will launch the **AI Resume Screening Web App** in your browser.  

---

## 📊 How It Works  

1️⃣ **Upload a Resume** (PDF or Text).  
2️⃣ **AI Parses the Resume** – Extracts **structured information**.  
3️⃣ **Ask AI Questions** – Type job-related questions & get AI-driven insights.  
4️⃣ **Get Role Suitability Analysis** – AI compares resume skills against job requirements.  

---

## 🎯 Example Queries  

✔️ *Does this candidate have experience with Python?*  
✔️ *What certifications does the candidate have?*  
✔️ *Is this candidate suitable for a Data Scientist role?*  

---

## 📦 Deployment  

🔹 **Deploy on Streamlit Cloud**  
git push origin main

yaml
Copy
Edit

🔹 **Run on AWS / Azure**  
- Package into a **Docker container**.  
- Deploy as an **Azure Web App / AWS Lambda function**.  

---

## 🤝 Contribution  
Want to improve this project? Feel free to:  
- Open an **issue** 🛠️  
- Submit a **pull request** ✨  
- Share **feedback** 💡  

---

## 🌟 Show Your Support  
If you find this project helpful, **please ⭐ star the repository!** 😊  