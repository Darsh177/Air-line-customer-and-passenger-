import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import os
import threading  
warnings.filterwarnings('ignore')

class AirlineSatisfactionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Airline Satisfaction Predictor")
        self.root.geometry("900x700")
        self.root.configure(bg="#90D1CA")

        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=10, background="#096B68")
        self.style.configure("TLabel", font=("Helvetica", 12), background="#90D1CA")
        self.style.configure("TCombobox", font=("Helvetica", 12))
        self.style.map("TButton", background=[("active", "#129990")])

        title_frame = tk.Frame(self.root, bg="#096B68")
        title_frame.pack(fill="x", pady=(5, 10))
        title_label = tk.Label(title_frame, text="Airline Satisfaction Predictor", font=("Helvetica", 18, "bold"), fg="#129990", bg="#096B68", pady=10)
        title_label.pack()

        main_frame = tk.Frame(self.root, bg="#90D1CA")
        main_frame.pack(padx=20, pady=10, fill="both", expand=True)

        control_frame = tk.Frame(main_frame, bg="#FFFBDE", relief="raised", bd=2, highlightbackground="#90D1CA")
        control_frame.pack(side="left", padx=10, pady=10, fill="both", expand=False)
        control_frame.config(width=300)
#بيحمل الملف  لم بندوس على الزرار 
        self.load_btn = ttk.Button(control_frame, text="Load Train & Test CSV", command=self.load_data, style="TButton")
        self.load_btn.pack(pady=10, padx=10, fill="x")
# ده الي بنختار منه النماذج 
        ttk.Label(control_frame, text="Select EDA Plot:", style="TLabel").pack(pady=(10, 5), padx=10)
        self.eda_var = tk.StringVar(value="Satisfaction Distribution")

        self.eda_dropdown = ttk.Combobox(control_frame, textvariable=self.eda_var, values=["Satisfaction Distribution", 
                                                                                           "Correlation Matrix",
                                                                                             "Satisfaction by Class"]
                                                                                             , state="readonly",
                                                                                               style="TCombobox", width=25)
        
        self.eda_dropdown.pack(pady=10, padx=10, fill="x")

        #ده الزرار الي بيعرض الجراف
        self.plot_btn = ttk.Button(control_frame, text="Show Plot", command=self.show_plot, style="TButton")
        self.plot_btn.pack(pady=10, padx=10, fill="x")

        ttk.Label(control_frame, text="Select Model:", style="TLabel").pack(pady=(10, 5), padx=10)
        self.model_var = tk.StringVar(value="Random Forest")
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': LinearSVC(C=1.0, random_state=42, max_iter=10000),  # تغيير إلى LinearSVC
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        self.model_dropdown = ttk.Combobox(control_frame, textvariable=self.model_var, values=list(self.models.keys()), state="readonly", style="TCombobox", width=25)
        self.model_dropdown.pack(pady=5, padx=10, fill="x")
        self.train_btn = ttk.Button(control_frame, text="Train & Evaluate Model", command=self.start_training_thread, style="TButton")  # تغيير لاستخدام threading
        self.train_btn.pack(pady=5, padx=10, fill="x")

        ttk.Label(control_frame, text="Enter Passenger Data:", style="TLabel").pack(pady=(20, 5), padx=10)
        self.input_frame = tk.Frame(control_frame, bg="#FFFBDE")
        self.input_frame.pack(pady=5, padx=10, fill="x")

        self.input_fields = {}
        self.input_vars = {}
        features = [("Gender", ["Male", "Female"]), 
                    ("Customer Type", ["Loyal Customer", "disloyal Customer"]), 
                    ("Type of Travel", ["Personal Travel", "Business travel"]), 
                    ("Class", ["Eco", "Eco Plus", "Business"]), 
                    ("Age", "numerical"), 
                    ("Flight Distance", "numerical"), 
                    ("Inflight wifi service", "numerical"), 
                    ("Ease of Online booking", "numerical"), 
                    ("Seat comfort", "numerical"), 
                    ("On-board service", "numerical")]
        for i, (feature, dtype) in enumerate(features):
            ttk.Label(self.input_frame, text=f"{feature}:", style="TLabel", background="#FFFBDE").grid(row=i, column=0, padx=5, pady=2, sticky="w")
            self.input_vars[feature] = tk.StringVar()
            if dtype != "numerical":
                self.input_fields[feature] = ttk.Combobox(self.input_frame, textvariable=self.input_vars[feature],
                                                           values=dtype, state="readonly", width=20)
                self.input_fields[feature].set(f"Select {feature} (e.g., {dtype[0]})")

            else:
                self.input_fields[feature] = tk.Entry(self.input_frame, textvariable=self.input_vars[feature], width=20)
                self.input_fields[feature].insert(0, f"Enter {feature} (e.g., {'30' if feature == 'Age' else '1000' if feature == 'Flight Distance' else '0-5'})")
                self.input_fields[feature].bind("<FocusIn>", lambda event, f=feature: self.clear_placeholder(event, f))
                self.input_fields[feature].bind("<FocusOut>", lambda event, f=feature: self.restore_placeholder(event, f))
            self.input_fields[feature].grid(row=i, column=1, padx=5, pady=2, sticky="w")

        self.predict_btn = ttk.Button(control_frame, text="Predict Satisfaction", command=self.predict_satisfaction, style="TButton")
        self.predict_btn.pack(pady=(10, 5), padx=10, fill="x")

        self.clear_btn = ttk.Button(control_frame, text="Clear Results & Plot", command=self.clear_output, style="TButton")
        self.clear_btn.pack(pady=(10, 5), padx=10, fill="x")

        result_frame = tk.Frame(main_frame, bg="#90D1CA")
        result_frame.pack(side="right", padx=10, pady=10, fill="both", expand=True)

        ttk.Label(result_frame, text="Results:", style="TLabel").pack(anchor="w")
        self.result_text = tk.Text(result_frame, height=5, width=50, font=("Helvetica", 11), bg="#FFFBDE", fg="#096B68")
        self.result_text.pack(pady=5, fill="x")

        plot_frame = tk.Frame(result_frame, bg="#FFFBDE", relief="sunken", bd=2)
        plot_frame.pack(pady=10, fill="both", expand=True)
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.train_df = None
        self.test_df = None
        self.results = {}
        self.label_encoders = {}
        self.scaler = None
        self.imputer = None
        self.trained_model = None
        self.feature_names = []
        self.training = False  

    def clear_placeholder(self, event, feature):
        entry = self.input_fields[feature]
        placeholder = f"Enter {feature} (e.g., {'30' if feature == 'Age' else '1000' if feature == 'Flight Distance' else '0-5'})"
        if entry.get() == placeholder:
            entry.delete(0, tk.END)

    def restore_placeholder(self, event, feature):
        entry = self.input_fields[feature]
        placeholder = f"Enter {feature} (e.g., {'30' if feature == 'Age' else '1000' if feature == 'Flight Distance' else '0-5'})"
        if not entry.get():
            entry.insert(0, placeholder)

    def load_data(self):
        try:
            train_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if not train_file:
                return
            self.train_df = pd.read_csv(train_file).drop(['Unnamed: 0','id'], axis=1).sample(frac=0.05, random_state=42)  # تقليل البيانات إلى 5%

            test_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if not test_file:
                return
            self.test_df = pd.read_csv(test_file).drop(['Unnamed: 0','id'], axis=1).sample(frac=0.05, random_state=42)  # تقليل البيانات إلى 5%

            self.preprocess_data()
            messagebox.showinfo("Success", "Data loaded and preprocessed successfully!")
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Train shape: {self.train_df.shape}\nTest shape: {self.test_df.shape}")
            self.result_text.update_idletasks()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {e}")

    def preprocess_data(self):
        combined_df = pd.concat([self.train_df, self.test_df], axis=0, ignore_index=True)

        numerical_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
        self.imputer = KNNImputer(n_neighbors=5)
        combined_df[numerical_cols] = self.imputer.fit_transform(combined_df[numerical_cols])

        categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        self.label_encoders = {}
        for col in categorical_cols:
            if col in combined_df.columns:
                self.label_encoders[col] = LabelEncoder()
                combined_df[col] = self.label_encoders[col].fit_transform(combined_df[col].astype(str))

        self.label_encoders['satisfaction'] = LabelEncoder()
        combined_df['satisfaction'] = self.label_encoders['satisfaction'].fit_transform(combined_df['satisfaction'])

        self.scaler = StandardScaler()
        combined_df[numerical_cols] = self.scaler.fit_transform(combined_df[numerical_cols])

        self.train_processed = combined_df.iloc[:len(self.train_df), :]
        self.test_processed = combined_df.iloc[len(self.train_df):, :]

        self.X_train = self.train_processed.drop(['satisfaction'], axis=1)
        self.y_train = self.train_processed['satisfaction']
        self.X_test = self.test_processed.drop(['satisfaction'], axis=1)
        self.y_test = self.test_processed['satisfaction']
        self.feature_names = self.X_train.columns.tolist()

    def show_plot(self):
        if self.train_df is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return

        self.ax.clear()
        plot_type = self.eda_var.get()

        try:
            if plot_type == "Satisfaction Distribution":
                sns.countplot(x='satisfaction', hue='satisfaction', data=self.train_df, ax=self.ax, palette="Blues", legend=False)
                self.ax.set_title("Satisfaction Distribution", fontsize=14, pad=15)
                self.ax.set_xlabel("Satisfaction", fontsize=12)
                self.ax.set_ylabel("Count", fontsize=12)

            elif plot_type == "Correlation Matrix":
                sns.heatmap(self.train_processed.corr(numeric_only=True), annot=False, cmap='coolwarm', ax=self.ax)
                self.ax.set_title("Correlation Matrix", fontsize=14, pad=15)

            elif plot_type == "Satisfaction by Class":
                sns.countplot(x='Class', hue='satisfaction', data=self.train_df, ax=self.ax, palette="Set2")
                self.ax.set_title("Satisfaction by Class", fontsize=14, pad=15)
                self.ax.set_xlabel("Class", fontsize=12)
                self.ax.set_ylabel("Count", fontsize=12)

            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot: {e}")

    def start_training_thread(self):
        if self.training:
            messagebox.showwarning("Warning", "Training already in progress. Please wait!")
            return
        self.training = True
        self.train_btn.config(state="disabled")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Training model, please wait...\n")
        self.result_text.update_idletasks()
        thread = threading.Thread(target=self.train_model)
        thread.start()
        self.root.after(100, self.check_training_thread, thread)

    def check_training_thread(self, thread):
        if thread.is_alive():
            self.root.after(100, self.check_training_thread, thread)
        else:
            self.training = False
            self.train_btn.config(state="normal")
            self.result_text.insert(tk.END, "Training completed.\n")
            self.result_text.update_idletasks()
            
    def train_model(self):
        if self.train_processed is None:
            self.root.after(0, lambda: messagebox.showwarning("Warning", "Please load data first!"))
            return
        try:
            model_name = self.model_var.get()
            self.trained_model = self.models[model_name]

            self.trained_model.fit(self.X_train, self.y_train)
            y_pred = self.trained_model.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            cm = confusion_matrix(self.y_test, y_pred)

            self.root.after(0, lambda: self.result_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.result_text.insert(tk.END, f"Model: {model_name}\n"))
            self.root.after(0, lambda: self.result_text.insert(tk.END, f"Accuracy: {accuracy:.4f}\n"))
            self.root.after(0, lambda: self.result_text.insert(tk.END, f"F1-Score: {f1:.4f}\n"))
            self.root.after(0, lambda: self.result_text.insert(tk.END, f"Confusion Matrix:\n{cm}\n"))
            self.root.after(0, lambda: self.result_text.insert(tk.END, "Model trained successfully. Ready to predict!\n"))
            self.root.after(0, lambda: self.result_text.update_idletasks())

            self.root.after(0, lambda: self.ax.clear())
            self.root.after(0, lambda: sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=self.ax))
            self.root.after(0, lambda: self.ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, pad=15))
            self.root.after(0, lambda: self.ax.set_xlabel("Predicted", fontsize=12))
            self.root.after(0, lambda: self.ax.set_ylabel("Actual", fontsize=12))
            self.root.after(0, lambda: self.canvas.draw())
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to train model '{model_name}': {str(e)}\nCheck data or try adjusting model parameters."))

    def predict_satisfaction(self):
        if self.trained_model is None:
            messagebox.showwarning("Warning", "Please train a model first!")
            return

        try:
            input_data = {}
            for feature in self.input_vars:
                value = self.input_vars[feature].get().strip()
                placeholder = f"Enter {feature} (e.g., {'30' if feature == 'Age' else '1000' if feature == 'Flight Distance' else '0-5'})"
                select_placeholder = f"Select {feature} (e.g., {self.input_fields[feature]['values'][0]})" if feature in ['Gender', 'Customer Type', 'Type of Travel', 'Class'] else None
                if not value or value == select_placeholder or value == placeholder:
                    messagebox.showwarning("Warning", f"Please enter a valid value for {feature}!")
                    return
                input_data[feature] = value

            df_input = pd.DataFrame(columns=self.feature_names)

            categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
            numerical_cols = ['Age', 'Flight Distance', 'Inflight wifi service', 'Ease of Online booking', 'Seat comfort', 'On-board service']
            for feature in input_data:
                if feature in self.feature_names:
                    if feature in categorical_cols and feature in self.label_encoders:
                        if input_data[feature] not in self.label_encoders[feature].classes_:
                            messagebox.showerror("Error", f"Invalid value for {feature}. Use one of: {list(self.label_encoders[feature].classes_)}")
                            return
                        encoded_value = self.label_encoders[feature].transform([input_data[feature]])[0]
                        df_input.at[0, feature] = encoded_value
                    elif feature in numerical_cols:
                        df_input.at[0, feature] = float(input_data[feature])

            df_input = df_input.reindex(columns=self.feature_names, fill_value=0)
            numerical_cols_all = [col for col in self.feature_names if col in self.train_df.select_dtypes(include=['float64', 'int64']).columns]
            df_input[numerical_cols_all] = self.imputer.transform(df_input[numerical_cols_all])
            df_input[numerical_cols_all] = self.scaler.transform(df_input[numerical_cols_all])
            prediction = self.trained_model.predict(df_input)[0]
            prediction_label = self.label_encoders['satisfaction'].inverse_transform([prediction])[0]
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Prediction: {prediction_label}\n")
            self.result_text.insert(tk.END, f"Model used: {self.model_var.get()}\n")
            self.result_text.insert(tk.END, "Input Data:\n")
            for feature, value in input_data.items():
                self.result_text.insert(tk.END, f"{feature}: {value}\n")
            self.result_text.update_idletasks()

            self.ax.clear()
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {e}")

    def clear_output(self):
        self.result_text.delete(1.0, tk.END)
        self.result_text.update_idletasks()
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AirlineSatisfactionGUI(root)
    root.mainloop()
