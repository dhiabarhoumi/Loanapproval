# Loan Approval System: End-to-End Machine Learning Application 

![Azure Deployment](https://img.shields.io/badge/Deployed%20on-Microsoft%20Azure-blue?style=flat-square)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-Flask-orange?style=flat-square)

Welcome to the Loan Approval System, an end-to-end machine learning application designed to predict loan approval decisions based on user inputs. This project leverages the power of machine learning pipelines, modern deployment practices, and clean code principles.

---

##  Features

### 🛠️ **End-to-End ML Pipeline**
- **Data Ingestion**: Handles input data efficiently and ensures robust preprocessing.
- **Data Transformation**: Applies feature engineering techniques for model-ready datasets.
- **Model Training**: Includes hyperparameter tuning using **Grid Search** to select the optimal model.
- **Prediction Pipeline**: Uses the best-performing model to generate predictions.

### 🌐 **Deployment**
- Fully deployed on **Microsoft Azure**, ensuring scalability and accessibility.
- Integrated **CI/CD pipelines** with GitHub Actions for automated testing and deployment.

### 🖥️ **Interactive User Interface**
- A sleek and responsive web interface built using **Flask**, **HTML**, and **Bootstrap**.
- **Home Page**: Welcomes users with an overview of the application.
- **Form**: Asks essential questions to determine loan approval status.

### 🛡️ **Robust Utilities**
- **Custom Exception Handling**: Simplifies debugging with meaningful error messages.
- **Logging System**: Generates detailed logs to monitor application performance.
- **Reusable Utility Functions**: Includes modular functions like `load_object`, `save_object`, and `_evaluate_models` for streamlined workflows.

---

## 📊 Tech Stack

| Technology       | Usage                              |
|-------------------|------------------------------------|
| **Flask**         | Backend framework                 |
| **HTML & Bootstrap** | Frontend for the web interface    |
| **Python**        | Core programming language         |
| **scikit-learn**  | Model training and evaluation     |
| **Microsoft Azure** | Deployment and hosting platform  |
| **GitHub Actions** | CI/CD implementation             |

---

## 🎯 Key Highlights

- **Hyperparameter Optimization**: Achieved by implementing Grid Search to ensure the best model selection.
- **Azure Deployment**: Demonstrates experience in cloud computing and operationalizing machine learning applications.
- **CI/CD Pipelines**: Showcases familiarity with DevOps practices for seamless project integration and delivery.
- **Custom Debugging Tools**: Includes a tailored exception handler to accelerate issue resolution.

---

## 🛠️ How to Use

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/dhiabarhoumi/loanapproval.git
   cd loanapproval
2. **Install Dependencies**

   ```bash
    pip install -r requirements.txt
3. **Run Locally**

   ```bash
    python app.py
4. **Access the Application**
Open your browser and navigate to:
http://localhost:5000

---

## 🧪 CI/CD Workflow

This project integrates GitHub Actions for continuous integration and deployment:
- **Build**: Ensures dependencies and the application structure are correct.
- **Test**: Automatically runs all test cases.
- **Deploy**: Deploys the latest changes to **Microsoft Azure**.

The `.github/workflows` folder contains the YAML files defining the CI/CD pipeline.

---

## 📸 Screenshots

1. **Home Page**
   ![Home Page](screenshots/home_page.png)  
   *This screenshot displays the application's home page with the input form for loan details.*

2. **Prediction Result**
   ![Prediction Result](screenshots/prediction_result.png)  
   *This screenshot shows the results page with the loan approval decision.*

---

## 🤝 Connect & Contribute

This project is open for contributions! Contributions, issues, and feature requests are welcome.  

---

## 📧 Contact

If you have any questions or feedback, feel free to reach out:

- **Name**: Dhieddine Barhoumi  
- **Portfolio**: [dhieddinebarhoumi.me](https://dhieddinebarhoumi.me)  
- **Email**: [dhiabarhoumi2608@gmail.com](mailto:dhiabarhoumi2608@gmail.com)  
- **LinkedIn**: [linkedin.com/in/barhoumi-dhieddine/](https://www.linkedin.com/in/barhoumi-dhieddine/)
- 
I’m open to discussions, collaborations, and new opportunities!



