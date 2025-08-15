# 📄 Data Extraction using Machine Learning

## 🚀 Project Overview
This project focuses on automating the extraction of **structured data**—including **text, tables, and images**—from unstructured documents such as **PDFs, Word, and HTML files**.  
The system is designed with a **robust frontend and backend architecture** to ensure smooth user interaction, secure data handling, and scalable high-performance processing.  

By leveraging **advanced machine learning techniques** such as **Mask R-CNN, OCR (Tesseract), and Layout Analysis models**, the application can accurately detect, classify, and convert document components into structured formats for further use.  

👉 This solution **reduces manual effort**, lowers operational costs, and minimizes human errors in **document-heavy workflows**.

---

## ✨ Key Features
- 🔹 Automated extraction of **text, tables, and images**  
- 🔹 Multi-format support – **PDF, Word, HTML, Images**  
- 🔹 **Mask R-CNN & LayoutParser** for document segmentation  
- 🔹 **OCR (Tesseract)** for accurate text recognition  
- 🔹 **Frontend (React)** for document upload, preview, and review  
- 🔹 **Backend (Django REST API)** for ML-powered processing  
- 🔹 Export structured data as **JSON / CSV / Text**  
- 🔹 **Role-based access control** & secure data handling  

---

## 🏗️ System Architecture
### 1. Frontend (ReactJS)
- Uploads documents (**PDF, Word, HTML**)  
- Displays extracted **text, tables, and images**  
- Provides **download/export** options  

### 2. Backend (Django + DRF)
- API endpoints for file upload & processing  
- Integrates **Mask R-CNN, LayoutParser, Tesseract OCR**  
- Converts content into structured formats  

### 3. Machine Learning Models
- **Mask R-CNN (Detectron2 + PubLayNet)** → Region detection  
- **OCR (Tesseract)** → Text recognition in detected blocks  
- **LayoutParser** → Semantic structure classification  

---

## 🛠️ Tech Stack
- **Frontend**: ReactJS, HTML, CSS, JavaScript  
- **Backend**: Django REST Framework, Python  
- **Machine Learning**: Detectron2, Mask R-CNN, LayoutParser, Tesseract OCR, OpenCV  
- **Database**: PostgreSQL / MySQL  
- **Deployment Tools**: Docker, VS Code, GitHub Actions  

---

## 📂 Modules of the Project
- **Data Collection Module** – Uploads documents (PDF, Word, HTML, Images)  
- **Data Preprocessing Module** – Cleans, transforms, and prepares input data  
- **Data Storage Module** – Saves extracted data in structured formats  
- **Security & Access Control Module** – Role-based authentication & encryption  
- **Data Analysis Module** – Provides structured outputs for dashboards & ML models  

---

## 📊 Example Workflow
1. User uploads a **PDF document**  
2. Backend converts the document into **images (per page)**  
3. **Mask R-CNN + LayoutParser** detect text, tables, and figures  
4. **OCR** extracts textual data from detected regions  
5. Extracted results are saved as:  
   - **Text/** → Extracted paragraphs  
   - **Tables/** → Extracted tables in CSV/JSON  
   - **Images/** → Extracted figures & images  

---

## ⚡ Results & Screenshots
✅ Text successfully extracted from PDF files  
✅ Tables parsed into structured **CSV format**  
✅ Figures/Images saved in **separate folders**  

*(You can add screenshots in your repo, e.g. `![UI Screenshot](screenshots/ui.png)`)*  

---

## 🔮 Future Enhancements
- 🌐 Multi-language OCR support  
- 🤖 Integration with **chatbots** using extracted data  
- ☁️ Cloud-based processing for **real-time scalability**  
- 📊 Advanced analytics dashboards with **Power BI / Tableau**  

---
