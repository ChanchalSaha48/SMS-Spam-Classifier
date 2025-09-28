# SMS Spam Classifier

# Dataset
 -[SMS Spam Collection Dataset(UCI/kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
 
 -5,574 **SMS** message labeled as **ham**(not spam) or **spam**

---

# Day1- Baseline Model: TF-IDF +Naive Bayes

- **Goal**: Build a very first baseline ML pipeline
- **Approach**:
     - Preprocess text -> TF-IDF feature
     - Train **Naive Bayes** classifier
- **Why Naive Bayes?**
     - Extremely fast
     - Works well for text classification
       
-**Result**: Got a simple but strong baseline accuracy.

**Code**: [day1_baseline_model.ipynb](https://github.com/ChanchalSaha48/SMS-Spam-Classifier/blob/main/day1_baseline_model.ipynb)


---

# Day2-  TF-IDF + Logistic Regression ( with n-grams)

- **Goal**: Compare Logistic Regression with Naive Bayes and test n-gram ranges.

- **Approach**:
     - Use ngram_range=(1,1),(1,2),(1,3)
     - Train **Logistic Regression** model
     - Compare Performance

- **Why n-grams?**
     - (1,1)-> single words
     - (1,2)-> word pairs like "free entry" ,"call now"
     - (1,3)-> triples, captures more context but may overfit

-**Result**: 

		n-gram Range  | Accuracy
		    (1,1)     | 96.86%
		    (1,2)     | 97.22%
		    (1,3)     | 97.40% 
			
**Code**: [day2_logreg_ngrams.ipynb](https://github.com/ChanchalSaha48/SMS-Spam-Classifier/blob/main/day_logreg_ngrams.ipynb)

---

# Day3- Hyperparmeter Tuning ( Logistic Regression & SVM )

- **Goal**: Improve model performence using GridSearchCV
  
- **Approch**:
  - perform **hyperparameter tuning** on **Logistic Regression** and **SVM**
  - Search over parameters like C, penalty, solver(for **Logistic Regression**) and C, gamma, kernel ( for **SVM** )
  - Evaluate models with **F1-Score**, **Classification Report** and **Confusion Matrix**
   
-**Why Hyperparameter Tuning?**
 - Default paramters may not be optimal
 - Proper tuing gives a more **generlized model**
 - **SVM** and **Logistic Regression** often benifit from tuning **C, penalty and kernel**

-**Results**:

                Model              |        Best Params                |  Accuracy
			 Logistic Regression   | C: 5, penalty: l2, solver: saga   | 97.85%
			 SVM                   | C: 10, gamma: 0.1, kernel: rbf    | 98.30%
			 
**Code**: [day3_hyperameter_tuning.ipynb](https://github.com/ChanchalSaha48/SMS-Spam-Classifier/blob/main/day3_hyperparameter_tuning.ipynb)

---

## Requirements 

pip install pandas scikit-learn matplotlib seaborn
