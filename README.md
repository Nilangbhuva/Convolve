# Development of a Behaviour Scoring Model for Credit Card Customers

> **Convolve 3.0: A Pan IIT AI/ML Hackathon**
>
> Team: deepdblm
>
> * Deep Das
> * Nilang Bhuva
> * Archit Savaliya
>
> **Institution:** Department Of Artificial Intelligence, Sardar Vallabhbhai National Institute of Technology (B.Tech 2nd Year)

---

## 📌 Project Objective

The primary objective of this project is to predict the probability of default for existing credit card customers. By developing a robust behavior scoring model, this tool aids in risk management, allowing financial institutions to identify high-risk profiles and prevent potential defaults.

---

## 📊 Dataset Overview

The project utilizes a comprehensive financial dataset categorized into several attribute types:

* **Development Dataset Size:** 96,806 records
* **Validation Dataset Size:** 41,792 records
* **Feature Categories:**
    * 48 On-us Attributes (credit limit related)
    * 664 Transaction Attributes (merchant data, counts, and values)
    * 452 Bureau Tradeline Attributes (product holdings, historical delinquencies)
    * 50 Bureau Enquiry Attributes (recent loan enquiries)

---

## ⚙️ Data Preprocessing Pipeline

To prepare the dataset for deep learning, we implemented a rigorous preprocessing pipeline:

* **Missing Values:** Dropped columns with over 80% missing data and used `SimpleImputer` to fill remaining missing values with the column mean.
* **Class Imbalance:** Addressed target variable imbalances using `SMOTETomek` to oversample the minority class and undersample the majority class.
* **Standardization:** Scaled features utilizing `StandardScaler`.

---

## 🧠 Model Architecture

We designed a custom Neural Network featuring 4 hidden layers and 6 convolutional layers to capture complex patterns in the financial data.

### Key Layers & Configurations:

* **Input Layer:** Reshaped features to fit `(input_dim, 1)` dimensions.
* **Convolutional Blocks (Conv1D):** Two Conv1D layers utilizing 64 filters, a kernel size of 3, and ReLU activation functions to extract local features.
* **Regularization:** Incorporated Batch Normalization to accelerate training and Dropout layers (rates of 0.5 and 0.4) to prevent overfitting.
* **Dense Layers:** Fully connected layers scaling down from 64, to 32, to 16 units, all utilizing ReLU activations.
* **Output Layer:** A single neuron with a Sigmoid activation function to output a probability value between 0 and 1.
* **Optimization:** Compiled using the Adam optimizer with a learning rate of 0.001 and Binary Crossentropy loss.

> **Risk Management Threshold:** The threshold for classifying a credit card holder as a defaulter was strictly set to 0.4 (instead of the standard 0.5) to conservatively prioritize the identification of potential defaulters.

---

## 📈 Performance Metrics

The model achieved highly reliable predictive metrics on the evaluation dataset:

* **Overall Accuracy:** 0.96
* **F1-Score:** 0.96
* **Recall (Defaulters):** 0.90

---

## 💡 Key Data Insights

During Exploratory Data Analysis (EDA), several critical behavioral patterns were identified:

* **Inquiry Load:** The majority of the population exhibits a low inquiry load; however, a highly skewed long tail indicates a small segment with extreme inquiry loads (>400), flagging high credit-seeking behavior and potential financial distress.
* **Spending Clusters:** Customer segmentation revealed distinct tiers, notably "Cluster 2", which represents high-value customers with significantly higher average spends compared to other groups.
* **Bureau Inquiry Risk Zones:** Analysis showed that the vast majority of assessed individuals fall safely into the "Low Risk" category, indicating a generally stable population profile.

---

## 📄 Full Documentation

See the full methodology, pipeline diagrams, and code documentation in [Documentation.pdf](Documentation.pdf).
