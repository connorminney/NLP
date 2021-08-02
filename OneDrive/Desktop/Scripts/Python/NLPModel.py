# Import dependencies
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk import FreqDist
from nltk import classify
from autocorrect import Speller
from random import shuffle
import time
import datetime
from PyQt5.QtWidgets import * 
from PyQt5 import QtCore, QtGui, QtWidgets 
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
import sys 
import os
from pathlib import Path
from glob import glob
from PIL import Image
import pickle


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.realpath('__file__'))
    return os.path.join(base_path, relative_path)

# Link the environment to the platforms foler (path will vary by user!) - this is required for pyinstaller to work #
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = resource_path('./driver/platforms')

# Create variable to identify the download folder path regardless of who the user is #
path_to_download_folder = str(os.path.join(Path.home(), "Downloads"))
        
# Create a 'Reports' folder within the downloads folder - this is where the downloads will go #
NLPpath = path_to_download_folder + '\\Text Analysis' 
if not os.path.exists(NLPpath):
    os.makedirs(NLPpath)

#=========================================================================================================================#
# Main function where work is performed #  
#=========================================================================================================================#  


# This is the function that is tied to the "Run Macro" button. Basically the full script is housed here #
class part_2(QThread):
    
    # This variable is required for the loading bar. Just make sure it stays here. #
    countChanged = pyqtSignal(int)
    Updates = pyqtSignal(str)
    
    def run(self):
        
        # This is for the progress bar. Do not alter.
        count = 10
        self.countChanged.emit(count)
        self.Updates.emit('Scanning File...')
        
        # Load in the dataframes with the relevant field names #
        NLP = pd.read_csv(NLPpath + '\\NLP.csv')
        print(NLP)
        
        # Extract the credtials/vendors from the dataframes #
        # Specify which column contains the text and which column contains the classifier #
        training_path = NLP.loc[0, 'FILE']
        TEXT_FIELD = NLP.loc[0, 'TEXT']
        CLASSIFIER_FIELD = NLP.loc[0, 'CLASSIFIER']
        testing_path = NLP.loc[0,'TEST FILE']
        
        # Delete the files from the file path #
        # os.remove(NLPpath + '\\NLP.csv')     
        
        print('file read and removed')
            
        # Automated model training and testing
        ''' Convert the text column to all lower case values and remove periods and commas.
            Also, assign standardized column names to the text and classifier fields so that
            they can be used by the script. Lastly, convert the classifier values to a binary
            format where positives are a 1 and negatives are a 0. '''
        
        # Load the training data 
        try:
            training_data = pd.read_excel(training_path)
        except:
            training_data = pd.read_csv(training_path)
            
        print('training loaded')
            
        # Create a list of the different types of classifiers
        values = list(training_data[CLASSIFIER_FIELD].unique())
        
        print(values)
        
        # Import spellcheck function #
        spell = Speller()
#        spell = Speller(lang='en')

        # Classify punctuations to remove #
        punctuations = '|'.join([',', "'",'#','%','^','&'])
            
        # Define a text normalization function to remove punctuation/numbers, drop everything to lowercase, and correct spelling
        def normalize_text(dataset):
            # Remove the punctuations (had to break it up because the script wasn't cooperating #
            count = 20
            self.countChanged.emit(count)
            self.Updates.emit('Removing Punctuations')
#            print('Removing Punctuations...')
            dataset['Text_Details'] = dataset[TEXT_FIELD].astype(str)
            dataset['Text_Details'] = dataset['Text_Details'].str.lower().str.replace(punctuations, '')
            dataset['Text_Details'] = dataset['Text_Details'].str.replace('.', ' ').str.replace('/', ' ').str.replace('+', ' ').str.replace(':', ' ').str.replace('=', ' ')
            dataset['Text_Details'] = dataset['Text_Details'].str.replace("_", '').str.replace("@", ' ').str.replace("?", ' ').str.replace("*", '').str.replace("$", '')
            dataset['Text_Details'] = dataset['Text_Details'].str.replace("!", ' ').str.replace("-", ' ').str.replace('"', '').str.replace(';', ' ')
            
            # Remove all of the numeric values
            count = 30
            self.countChanged.emit(count)
            self.Updates.emit('Formatting Text')
            for i in range(len(dataset)):
                try:
                    dataset.loc[i, 'Text_Details'] = re.sub(r'[0-9]', '', dataset.loc[i, 'Text_Details'])
                    self.Updates.emit('Formatting ' + str(i))
                    print('Formatting ' + str(i))
                except:
                    pass
            
            
            count = 40
            self.countChanged.emit(count)
            self.Updates.emit('Correcting Spelling')
            # Run spell correction 
            for i in range(len(dataset)):
                try:
                    dataset.loc[i, 'Text_Details'] = spell(dataset.loc[i, 'Text_Details'])
                    # self.Updates.emit('Spell Checking ' + str(i))
                    print('Spell Checking ' + str(i))
                except:
                    pass
            
            
            # Drop any null values and reindex the data
            dataset = dataset.dropna().reset_index()
        
        # Run the normalization function over the dataset
        normalize_text(training_data)
            
        # Assign numeric values to the text variables
        training_data['Exceptions'] = training_data[CLASSIFIER_FIELD]
        for i in range(len(training_data)):
            for j in range(len(values)):
                if training_data.loc[i, 'Exceptions'] == values[j]:
                    training_data.loc[i, 'Exceptions'] = j
                    
        ''' Split the dataset into lists of lists where all records that we want to identify
            are placed into different lists based on their classifier. These dataframes are
            then stored in a dictionary where the classifier is used as the key. '''
        dct = {}
        for i in values:
            dct['%s' % i] = list(training_data[training_data[CLASSIFIER_FIELD] == i]['Text_Details'])
        
        #======================================================================================================================#
        ''' This is where the model is trained '''
         
        # Establish stopwords #
        stop_words = stopwords.words('english')
        
        ''' Create a 'remove_noise' function to clear out the 
            stopwords & irrelevant text. Additionally, it uses
            pos_tag() and lemmatizing in order to identify the
            context of the words and assign them an appropriate 
            root.'''
        
        # Lemmatize the words 
        def remove_noise(text_tokens):
            
            count = 50
            self.countChanged.emit(count)
            self.Updates.emit('Removing Stopwords & Lemmatizing ' + str(i))
            
            cleaned_tokens = []
            
            for token, tag in pos_tag(text_tokens):
                
                # Set a position label that will be used by the lemmatizer
                if tag.startswith("NN"):
                    position = 'n'
                elif tag.startswith('VB'):
                    position = 'v'
                else:
                    position = 'a'
                
                # Lemmatize the tokens
                lemmatizer = WordNetLemmatizer()
                token = lemmatizer.lemmatize(token, pos = position)
                
                # Drop the stopwords
                if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                    cleaned_tokens.append(token.lower())
            
            return cleaned_tokens
                 
        
        ''' Convert the complaints in the dictionary to lists of tokens. We now
            have all of the complaints stored in a dictionary with 3 levels of lists.
            The first level contains all complaints broken into lists by their classifier.
            The second level breaks each complaint into its own list item.
            The third level breaks each word in the complaints into a list of tokens. '''
        for j in range(len(values)):
            tokenized_sents = [word_tokenize(i) for i in list(dct.values())[j]]
            text_tokens = []
            for k in tokenized_sents:
                text_tokens.append(k)
                dct['%s' % values[j]] = text_tokens
            
        
        ''' Lemmatize the tokens and remove stopwords '''
        for j in range(len(values)):
            i = 0
            cleaned_tokens_list = []
            for tokens in list(dct.values())[j]:
                cleaned_tokens_list.append(remove_noise(tokens))
                i = i + 1
                print('Lemmatizing ' + str(i))
            dct['%s' % values[j]] = cleaned_tokens_list
        
        
        ''' Define a function that gathers all of the words in all of
            the complaints and combines them into a single object. This
            is so that we can determine word density, which would
            not be practical on individual complaints because they are
            too short. THIS HAS BEEN COMMENTED OUT BECAUSE IT IS 
            ONLY FOR REVIEW/ASSURANCE PURPOSES.'''  
#        def get_all_words(cleaned_tokens_list):
#            for tokens in cleaned_tokens_list:
#                for token in tokens:
#                    yield token
#        
#        all_group_words = get_all_words(list(dct.values())[0])
#        freq_dist = FreqDist(all_group_words)
        
        ## Define a function to convert tokens to dictionary items #
        def get_text_for_model(cleaned_tokens_list):
            for text_tokens in cleaned_tokens_list:
                yield dict([token, True] for token in text_tokens)
        
        ''' Convert the dictionaries into lists of tuples with identifier tags for
            the record type tied to each value in the dictionary, then append everything
            to a list of tuples that will be used in the model. '''
        full_dataset = []
        for i in range(len(values)):
            dct['%s' % values[i]] = get_text_for_model(list(dct.values())[i])
            dataset = [(text_dict, list(dct.keys())[i])
                        for text_dict in list(dct.values())[i]]
            full_dataset = full_dataset + dataset
        
        # Shuffle the combined dataset #
        shuffle(full_dataset)
        
        # Split the dataset into training and test sets #
        train_data = full_dataset[:int(round(len(full_dataset)*(9/10),0))]
        test_data = full_dataset[int(round(len(full_dataset)*(1/10),0)):]
        
        print('Training Model')
        count = 60
        self.countChanged.emit(count)
        self.Updates.emit('Training Model (takes time)')
        ''' Plug the training data into a classifier. The following classifier types are available:
            1 - ConditionalExponentialClassifier (~95% accurate)
            2 - DecisionTreeClassifier (~99% accurate)
            3 - MaxentClassifier (~95% accurate)
            4 - NaiveBayesClassifier (~50% accurate)'''
        classifier = nltk.MaxentClassifier.train(train_data)
        
        # Print some summary information about the test #
        print("Accuracy is:", classify.accuracy(classifier, test_data))
        accuracy = round(classify.accuracy(classifier, test_data)*100, 2)
        accuracy_df = pd.DataFrame()
        accuracy_df.loc[0, 'Accuracy'] = accuracy
        accuracy_df.to_csv(NLPpath + '\\NLP.csv', index = False)
        
        # Export the trained model #
        model = open(NLPpath + '\\Trained Model.pickle', 'wb')
        pickle.dump(classifier, model)
        model.close()
        
        try:
            print(classifier.show_most_informative_features(100))
        except:
            pass
        
        ''' This is just for testing the model over user-defined complaints to 
            determine accuracy. '''
#        custom_complaint = 'I am very displeased. You will be hearing from my attorney.'
#        custom_tokens = remove_noise(word_tokenize(custom_complaint))
#        print(custom_tokens)
#        print(classifier.classify(dict([token, True] for token in custom_tokens)))
        
        #========================================================================================#
        ''' The rest of the script is where the actual testing occurs. Everything up to this point
            was training the model. For the next step, you will upload the document that you wish
            to test. '''
        #========================================================================================#
        # Load the test data 
        try:
            test_data = pd.read_excel(testing_path)
        except:
            test_data = pd.read_csv(testing_path)

        # Define a text normalization function to remove punctuation/numbers, drop everything to lowercase, and correct spelling
        def normalize_test_text(dataset2):

#            print('Removing Punctuations...')
            dataset2['Text_Details'] = dataset2[TEXT_FIELD].astype(str)
            dataset2['Text_Details'] = dataset2['Text_Details'].str.lower().str.replace(punctuations, '')
            dataset2['Text_Details'] = dataset2['Text_Details'].str.replace('.', ' ').str.replace('/', ' ').str.replace('+', ' ').str.replace(':', ' ').str.replace('=', ' ')
            dataset2['Text_Details'] = dataset2['Text_Details'].str.replace("_", '').str.replace("@", ' ').str.replace("?", ' ').str.replace("*", '').str.replace("$", '')
            dataset2['Text_Details'] = dataset2['Text_Details'].str.replace("!", ' ').str.replace("-", ' ').str.replace('"', '').str.replace(';', ' ')
            
            # Remove all of the numeric values
            for i in range(len(dataset2)):
                try:
                    dataset2.loc[i, 'Text_Details'] = re.sub(r'[0-9]', '', dataset2.loc[i, 'Text_Details'])
                    print('Formatting ' + str(i))
                except:
                    pass
                
            # Run spell correction 
            for i in range(len(dataset2)):
                try:
                    dataset2.loc[i, 'Text_Details'] = spell(dataset2.loc[i, 'Text_Details'])
                    print('Spell Checking ' + str(i))
                except:
                    pass
                
            # Drop any null values and reindex the data
            dataset2 = dataset2.dropna().reset_index()
        
        
        count = 70
        self.countChanged.emit(count)
        self.Updates.emit('Normalizing Test Data')
        # Normalize the test data text 
        normalize_test_text(test_data)
        
        count = 80
        self.countChanged.emit(count)
        self.Updates.emit('Predicting Values')
        # Run the analysis
        for i in range(len(test_data)):
            tokens = remove_noise(word_tokenize(test_data.loc[i, 'Text_Details']))
            test_data.loc[i, 'Predicted_Value'] = classifier.classify(dict([token, True] for token in tokens))
            print(i)
        
        count = 90
        self.countChanged.emit(count)
        self.Updates.emit('Exporting Results')
        
        # Export the test data with the predicted values
        test_data.to_csv(NLPpath + '\\Predicted Values.csv')
        count = 100
        self.countChanged.emit(count)
        self.Updates.emit('Complete!')
    
    def stop(self):
        self.Updates.emit(' Cancelled')
        count = 0
        self.countChanged.emit(count)
        self.threadactive = False
        self.disconnect()
#=========================================================================================================================#
# End of Third Party Risk Management testing script. The rest builds the GUI.
#=========================================================================================================================#

# Build the user interface  #
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 600)
        
        # Start button widget #
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(460, 20, 75, 50))
        self.pushButton.setObjectName("pushButton")
        
        # Button to run analysis
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(200, 300, 75, 50))
        self.pushButton2.setObjectName("pushButton2")
        self.pushButton2.hide()
        
        # Button to quit analysis
        self.QuitButton = QtWidgets.QPushButton(self.centralwidget)
        self.QuitButton.setGeometry(QtCore.QRect(285, 300, 75, 50))
        self.QuitButton.setObjectName("QuitButton")
        self.QuitButton.hide()
              
        # load file label #
        self.File = QtWidgets.QLabel(self.centralwidget)
        self.File.setGeometry(QtCore.QRect(20, 20, 110, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.File.setFont(font)
        self.File.setObjectName("File")
        # load file input #
        self.FileInput = QtWidgets.QLineEdit(self.centralwidget)
        self.FileInput.setGeometry(QtCore.QRect(140, 20, 300, 50))
        self.FileInput.setObjectName("FileInput")
        
        # Text field label
        self.TextListLabel = QtWidgets.QLabel(self.centralwidget)
        self.TextListLabel.setGeometry(QtCore.QRect(20, 90, 110, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.TextListLabel.setFont(font)
        self.TextListLabel.setObjectName("TextListLabel")
        self.TextListLabel.hide()
        # text field widget #
        self.TextList = QtWidgets.QComboBox(self.centralwidget)
        self.TextList.setGeometry(QtCore.QRect(150, 90, 290, 50))
        self.TextList.setObjectName("TextList")
        self.TextList.hide()
        
        # classifier label
        self.ClassifierLabel = QtWidgets.QLabel(self.centralwidget)
        self.ClassifierLabel.setGeometry(QtCore.QRect(20, 160, 110, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.ClassifierLabel.setFont(font)
        self.ClassifierLabel.setObjectName("ClassifierLabel")
        self.ClassifierLabel.hide()
        # classifier widget #
        self.Classifier = QtWidgets.QComboBox(self.centralwidget)
        self.Classifier.setGeometry(QtCore.QRect(150, 160, 290, 50))
        self.Classifier.setObjectName("Classifier")
        self.Classifier.hide()
        
        # load testing file label #
        self.File2 = QtWidgets.QLabel(self.centralwidget)
        self.File2.setGeometry(QtCore.QRect(20, 230, 110, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.File2.setFont(font)
        self.File2.setObjectName("File2")
        self.File2.hide()
        # load file input #
        self.File2Input = QtWidgets.QLineEdit(self.centralwidget)
        self.File2Input.setGeometry(QtCore.QRect(140, 230, 300, 50))
        self.File2Input.setObjectName("File2Input")
        self.File2Input.hide()
        
        # Create the progress bar #
        self.progress = QProgressBar(self.centralwidget)
        self.progress.setGeometry(30, 300, 450, 50)
        self.progress.setMaximum(100)
        self.progress.setVisible(False)
        

        # Standard stuff that should be included in every GUI
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 451, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Load File"))
        self.pushButton2.setText(_translate("MainWindow", "Run Analysis"))
        self.QuitButton.setText(_translate("MainWindow", "Cancel"))
        self.File.setText(_translate("MainWindow", "Training Filepath"))
        self.TextListLabel.setText(_translate("MainWindow", "Text to Analyze"))
        self.ClassifierLabel.setText(_translate("MainWindow", "Text Classifier"))
        self.File2.setText(_translate("MainWindow", "Testing Filepath"))
            
        # this line was added to connect the run button to the macro created below #
        self.pushButton.clicked.connect(self.Inputs)
        self.pushButton2.clicked.connect(self.SecondScriptRun)
        self.QuitButton.clicked.connect(self.killthread)

        
    def Inputs(self):
        
        # Convert the filepaths to strings that can be used
        path = self.FileInput.text()
        path = path.replace("\\", "/")

        test_path = self.File2Input.text()
        test_path = path.replace("\\", "/")
        
        # Read the file in and convert the column names to a list
        try:
            file = pd.read_excel(path)
        except:
            file = pd.read_csv(path)
        columns = list(file)
        
        # Add the column names and show the dropdown lists
        self.TextList.addItems(columns)
        self.TextList.show()
        self.TextListLabel.show()
        
        self.Classifier.addItems(columns)
        self.Classifier.show()
        self.ClassifierLabel.show()
        
        # Show the textbox where the path to the testing file goes
        self.File2.show()
        self.File2Input.show()
        
        # Display the button used to run the full analysis
        self.pushButton2.show()
        self.QuitButton.show()
        
        
    def SecondScriptRun(self):
        
        # Convert the filepath to a string that can be used
        path = self.FileInput.text()
        path = path.replace("\\", "/")
        # Convert the filepath to a string that can be used
        test_path = self.File2Input.text()
        test_path = test_path.replace("\\", "/")
        
        # Export the user inputs to a .csv file that will be read back in by the main loop #
        file_path = pd.DataFrame(columns = ['FILE'])
        file_path.loc[0,'FILE'] = path
        file_path.loc[0, 'TEXT'] = str(self.TextList.currentText())
        file_path.loc[0,'TEST FILE'] = test_path
        file_path.loc[0, 'CLASSIFIER'] = str(self.Classifier.currentText())
        file_path.to_csv(NLPpath + '\\NLP.csv', index = False)
        
        # Connect to the other thread and run #
        self.thread2 = part_2()
        self.thread2.start()
        
        # Hide original progress bar and show the new one
        self.progress.show()
        
        '''This is specifically for updating the progress bar as the script runs. Make sure it
            is connected to thread2.'''
        self.thread2.countChanged.connect(self.onCountChanged)
        self.thread2.Updates.connect(self.onUpdates)
        
        # Connect to the notification when finished running 
        self.thread2.finished.connect(self.notification)


 # Define a function that provides a notification when the SLA process is complete     
    def notification(self):
        time.sleep(5)
        
        accuracy = pd.read_csv(NLPpath + '\\NLP.csv')
        accuracy = accuracy.loc[0, 'Accuracy']
        # Send a message indicating what went wrong, if possible
        msg = QMessageBox()
        msg.setWindowTitle("Alert!")
        msg.setText('Analysis Complete. Your model is ' +str(accuracy) +'% accurate. You can find the full results here: \n ' + str(NLPpath) + '\\Predicted Values.csv \n')
#        msg.setInformativeText(errors)
        x = msg.exec_()
        
        # Delete the files from the file path #
        # try:
        #     os.remove(NLPpath + '\\NLP.csv')  
        # except:
        #     pass

############################################################################################################################################################
            
    # This is used by the Hiperos function to assign values to the progress bar #
    def onCountChanged(self, value):
        self.progress.setValue(value)
    def onUpdates(self, value):
        self.progress.setVisible(True)
        self.progress.setFormat(value)
    
    # Define a function to kill the script when the user clicks the 'cancel' button - tied to the 'stop' function in the access management script #
    def killthread(self):
        # The stop function is embedded in the script
        self.thread2.stop()
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
