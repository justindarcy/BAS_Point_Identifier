import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from textblob.classifiers import NaiveBayesClassifier
import pickle
from tkinter import Tk, Label, Button, RIGHT, LEFT, messagebox
from datetime import datetime


def select_file(import_title):
    """Get filepath of folder with excel docs from user."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=import_title)
    return(file_path)


def import_data(file_path):
    master_data = pd.read_excel(file_path)
    return(master_data)


def data_visulize(master_data):
    # print(master_data.head())
    # print(master_data.groupby('Simple Name').size())
    sns.countplot(master_data['Simple Name'], label="Count")
    plt.show()


def train_model(master_data):
    # use 20% of data to train machine learning model
    data_for_training = 0.2
    L = int(round(len(master_data) * data_for_training, 0))
    master_data = master_data[["BAS Point", "Simple Name"]]
    train = master_data[:L].values.tolist()
    test = master_data.ix[L + 1:].values.tolist()
    # python -m textblob.download_corpora    could be the link to fix errors at work: https://stackoverflow.com/questions/35861482/nltk-lookup-error
    # nltk.download()
    # nltk.download('punkt')
    cl = NaiveBayesClassifier(train)
    ml_accuracy = cl.accuracy(test)
    print("Model Accuracy:", ml_accuracy)
    return(cl, ml_accuracy)


def path_to_name(file_path):
    """Take the file paths and return the file names."""
    import os
    return(os.path.splitext(os.path.basename(file_path))[0])


def file_to_csv(output_data, name):
    # pick and assign title to the CSV output
    csv_title = str((datetime.now().strftime(
        "%Y.%m.%d_%H.%M%p") + " " + name + "_named.csv"))
    output_data.to_csv(csv_title, encoding='utf-8')
    messagebox.showinfo("Export Completed",
                        "File saved in python source folder")


def save_model(cl, file_name, ml_accuracy):
    file = open(datetime.now().strftime(
        "%H.%M%p") + " " + file_name + str(round(ml_accuracy, 2)) + '.pickle', 'wb')
    pickle.dump(cl, file)


def new_points_id(cl, input_data):
    output = []
    output_data = input_data
    for column in input_data.columns:
        output_data.loc[0, column] = cl.classify(column)
    return(output_data)


class ML_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Machine Learning BAS Point Name Tool")

        self.label = Label(
            master, text="Do you want to use an existing machine learning model or build a new one?")
        self.label.pack()

        self.build_button = Button(
            master, text="Build New Model", command=self.build)
        self.build_button.pack(side=RIGHT)

        self.load_button = Button(
            master, text="Load Existing", command=self.load)
        self.load_button.pack(side=LEFT)

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def build(self):
        print("BUILD a new model!")
        file_path = select_file(
            "Select training data for point ID. col_1=Simple name, col_2=BAS Name")
        master_data = import_data(file_path)
        cl, ml_accuracy = train_model(master_data)
        file_name = path_to_name(file_path)
        save_model(cl, file_name, ml_accuracy)
        data_visulize(master_data)

    def load(self):
        print("LOAD a Model!")
        cl_file_path = select_file("Select existing .pickle file")
        new_points_path = select_file("Select new points for id")
        new_input_data = import_data(new_points_path)
        ml_file = pickle.load(open(cl_file_path, 'rb'))
        output_data = new_points_id(ml_file, new_input_data)
        file_to_csv(output_data, path_to_name(new_points_path))


root = Tk()
my_gui = ML_GUI(root)
root.mainloop()
