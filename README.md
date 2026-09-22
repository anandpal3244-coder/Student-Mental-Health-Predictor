<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:43e97b&height=220&section=header&text=Mental%20Health%20Score%20Predictor&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Student%20Social%20Media%20%26%20Mental%20Health%20Impact&descAlignY=58&descSize=18" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Poppins&size=22&duration=3000&pause=1000&color=43E97B&center=true&vCenter=true&width=650&lines=Predicting+Student+Mental+Health+from+Social+Media+Habits;Built+with+Python+%2B+Scikit-learn+%2B+Streamlit;Random+Forest+Regression+%7C+R%C2%B2+%3D+0.877" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-38f9d7?style=for-the-badge)](LICENSE)

[![Repo Size](https://img.shields.io/github/repo-size/anandpal3244-coder/Student_Social_Media_And_Mental_Health_Impact?style=flat-square&color=43e97b)](.)
[![Last Commit](https://img.shields.io/github/last-commit/anandpal3244-coder/Student_Social_Media_And_Mental_Health_Impact?style=flat-square&color=38f9d7)](.)
![Visitors](https://api.visitorbadge.io/api/visitors?path=anandpal3244-coder%2FStudent_Social_Media_And_Mental_Health_Impact&label=Visitors&countColor=%2343e97b)

<a href="https://student-mental-health-predictor-xdcutxikw8qd8mwsqv8wc4.streamlit.app/"><b>Live Demo</b></a> ·
<a href="#-features"><b>Features</b></a> ·
<a href="#-installation--run-locally"><b>Installation</b></a> ·
<a href="#-model-performance"><b>Model Performance</b></a> ·
<a href="#-author"><b>Author</b></a>

</div>

<br/>

## 📌 About the Project

A machine-learning powered **Streamlit web app** that predicts a student's **Mental Health Score (0–10)** from their social media usage, sleep, study habits, physical activity and stress level — trained on a 5,000-student dataset and served through an interactive, animated UI with live gauges and charts.

> 🎯 **Goal:** Understand how digital habits and lifestyle factors relate to student well-being, and turn that analysis into a real, usable prediction tool.

<br/>

## 🚀 Live Demo

<div align="center">

**[👉 Try the app here](https://student-mental-health-predictor-xdcutxikw8qd8mwsqv8wc4.streamlit.app/)** &nbsp;*(replace with your deployed Streamlit Cloud link)*

<img src="https://raw.githubusercontent.com/anandpal3244-coder/Student_Social_Media_And_Mental_Health_Impact/main/assets/demo.gif" width="85%" alt="App demo"/>

*(Add a screen-recording GIF of your running app at `assets/demo.gif` — tools like [ScreenToGif](

https://github.com/user-attachments/assets/34ab3a59-754e-4a0d-8fa5-0f6cd74fb1cd

) work great for this)*

</div>

<br/>

## ✨ Features

| | |
|---|---|
| 🔮 **Live Prediction** | Enter a student's habits → get an instant Mental Health Score with an animated gauge |
| 📊 **Dataset Insights** | Upload the CSV for live, interactive histograms, box plots, scatter plots & correlation heatmaps |
| 🤖 **Model Comparison** | Real R² / MAE / RMSE metrics comparing Linear Regression vs Random Forest (default & tuned) |
| 🎨 **Modern UI** | Glassmorphism cards, animated gradient backgrounds, gauge charts, hover effects |
| ⚡ **Zero Setup for Users** | Deployed as a one-click web app — no installs needed to try it |

<br/>

## 🧠 How It Works

```mermaid
flowchart LR
    A[Student Inputs] --> B[Preprocessing Pipeline]
    B --> C[Log-transform + Scaling]
    B --> D[Ordinal Encode: Stress Level]
    B --> E[One-Hot Encode: Categorical]
    C --> F[Random Forest Regressor]
    D --> F
    E --> F
    F --> G[Mental Health Score 0-10]
```

<br/>

## 📈 Model Performance

| Model | R² (Test) | R² (Train) | MAE | RMSE |
|---|:---:|:---:|:---:|:---:|
| Linear Regression | 0.740 | 0.724 | 0.536 | 0.676 |
| **Random Forest (default)** ✅ | **0.877** | 0.981 | **0.348** | **0.464** |
| Random Forest (tuned) | 0.866 | 0.955 | 0.369 | 0.486 |

✅ **Deployed model:** Random Forest (default) — best balance of accuracy and generalization.

<br/>

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Git](https://img.shields.io/badge/-Git-F05032?style=flat-square&logo=git&logoColor=white)

</div>

<br/>

## 📁 Project Structure

```
mental_health_app/
│
├── app.py                     # Streamlit application (UI + prediction logic)
├── requirements.txt           # Python dependencies
├── Mental_Health_Model.pkl    # Trained Random Forest pipeline (joblib)
├── Student_Social_Media_And_Mental_Health_Impact.ipynb   # EDA + model training notebook
└── README.md                  # You are here
```

<br/>

## ⚙️ Installation & Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/anandpal3244-coder/Student_Social_Media_And_Mental_Health_Impact.git
cd Student_Social_Media_And_Mental_Health_Impact

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
#   Windows:
.venv\Scripts\activate
#   macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

<br/>

## 🌐 Deployment

Deployed free on **Streamlit Community Cloud**:

1. Push this repo to GitHub (includes `app.py`, `requirements.txt`, `Mental_Health_Model.pkl`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select this repo → branch `main` → main file `app.py` → **Deploy**

<br/>

## 🗺️ Roadmap

- [ ] Add SHAP-based feature-importance explanations per prediction
- [ ] Add a "compare two students" side-by-side view
- [ ] Multi-language support
- [ ] Downloadable PDF prediction report

<br/>

## 🤝 Contributing

Contributions, issues and feature requests are welcome!
Feel free to check the [issues page](../../issues) or open a pull request.

<br/>

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

<br/>

## 👤 Author

<div align="center">

### Anand Kumar
**Data Analyst** — Python · SQL · Power BI · Tableau · Streamlit

[![GitHub](https://img.shields.io/badge/GitHub-anandpal3244--coder-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/anandpal3244-coder)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Anand%20Pal-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anand-pal-6a657b393/)
[![Gmail](https://img.shields.io/badge/Email-anandpal3244%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:anandpal3244@gmail.com)

<br/>

<img src="https://github-readme-stats.vercel.app/api?username=anandpal3244-coder&show_icons=true&theme=radical&hide_border=true&count_private=true" width="48%" alt="GitHub Stats"/>
<img src="https://github-readme-streak-stats.herokuapp.com/?user=anandpal3244-coder&theme=radical&hide_border=true" width="48%" alt="GitHub Streak"/>

<br/><br/>

**Other Projects**

[![Career Prediction](https://img.shields.io/badge/🎓_Student_Career_Success_Prediction-Live_Demo-43e97b?style=flat-square)](https://student-career-success-prediction-n4xvvdute4gcwqjrwgyjdp.streamlit.app/)
[![Movie Recommender](https://img.shields.io/badge/🎬_Movie_Recommendation_System-Live_Demo-38f9d7?style=flat-square)](https://movie-recommender-jdts3fvmdqq3vril8uh8sd.streamlit.app/)
[![Retail Dashboard](https://img.shields.io/badge/📊_Retail_Sales_Dashboard-Tableau-f9d423?style=flat-square)](https://public.tableau.com/app/profile/anand.kumar1736/viz/RetailSalesPerformanceDashboard_17820255179240/RetailSalesPerformanceDashboard?publish=yes)
[![SQL Analysis](https://img.shields.io/badge/🎵_Music_Store_SQL_Analysis-Repo-ff5f6d?style=flat-square)](https://github.com/anandpal3244-coder/Music-Store-SQL-Analysis)

</div>

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:43e97b,100:0f0c29&height=120&section=footer" width="100%"/>
