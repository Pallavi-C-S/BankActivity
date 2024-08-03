from django.shortcuts import render
from .models import userdetails
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import numpy as np

# Create your views here.


def index(request):
    return render(request,'myapp/index.html')

def login(request):
    if request.method == "POST":
        username = request.POST.get('uname')
        password = request.POST.get('pwd')
        print(username, password)

        # Check if username and password match admin credentials
        if username == 'admin' and password == 'admin':
            request.session['uname']='admin'
            content = {
                'data1': 'admin'
            }
            return render(request, 'myapp/homepage.html', content)

        else:
            try:
                # Query the database for user details
                user = userdetails.objects.get(first_name=username, password=password)
                request.session['userid'] = user.id
                request.session['uname'] = user.first_name
                print(user.id)
                content={
                    'data1':user.first_name
                }
                return render(request, 'myapp/homepage.html',content)
            except userdetails.DoesNotExist:
                return render(request, 'myapp/login.html')
    return render(request,'myapp/login.html')

def register(request):
    if request.method == 'POST':
        first_name = request.POST['firstname']
        last_name = request.POST['lastname']
        emailid = request.POST['email']
        mobileno = request.POST['mobno']
        # username = request.POST['uname']
        password = request.POST['pwd']

        newuser = userdetails(first_name=first_name, last_name=last_name, emailid=emailid, password=password,phonenumber=mobileno)
        newuser.save()
        return render(request, "myapp/login.html", {})
    return render(request,'myapp/register.html')

def homepage(request):
    return render(request,'myapp/homepage.html')

def dataupload(request):
    df = pd.read_csv('bank.csv')
    res=df.shape[0]
    content={
        'data1':res
    }
    return render(request,'myapp/dataupload.html',content)

def createmodel(request):
    # Load data
    df = pd.read_csv('bank.csv')

    # Preprocess data
    # Convert categorical variables using LabelEncoder
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Define features and target
    X = df.drop('deposit', axis=1)
    y = df['deposit']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print('Confusion Matrix:\n', conf_matrix)

    # Plotting and saving the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    # plt.show()

    # Save the model and encoders
    dump(model, 'model.joblib')
    for col, le in label_encoders.items():
        dump(le, f'{col}_encoder.joblib')

    content={
        'data1':accuracy,
        'data2':precision,
        'data3':recall,
        'data4':f1,
        'data5':conf_matrix,

    }

    print("Model, encoders, and confusion matrix have been saved successfully.")
    return render(request,'myapp/createmodel.html',content)

def predictdata(request):
    # Load the model and encoders
    if request.method=="POST":

        model = load('model.joblib')
        encoders = {col: load(f'{col}_encoder.joblib') for col in
                    ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']}

        # Function to encode input data
        def encode_input(input_data):
            for col, encoder in encoders.items():
                input_data[col] = encoder.transform([input_data[col]])[0]
            return input_data

        # Example user input
        user_input = {
            'age': request.POST['age'],
            'job': request.POST['job'],
            'marital': request.POST['marital'],
            'education': request.POST['education'],
            'default': request.POST['default'],
            'balance': request.POST['balance'],
            'housing': request.POST['housing'],
            'loan': request.POST['loan'],
            'contact': request.POST['contact'],
            'day': request.POST['day'],
            'month': request.POST['month'],
            'duration': request.POST['duration'],
            'campaign': request.POST['campaign'],
            'pdays': request.POST['pdays'],
            'previous': request.POST['previous'],
            'poutcome': request.POST['poutcome']
        }

        # Encode user input
        encoded_input = np.array([list(encode_input(user_input).values())])

        # Predict
        prediction = model.predict(encoded_input)
        print('Prediction:', 'Deposit' if prediction[0] == 1 else 'No Deposit')
        if prediction[0] == 1:
            res='Deposit'
        else:
            res='No Deposit'
        content={
            'data1':'Prediction:'+res
        }
        return render(request, 'myapp/predictdata.html',content)
    return render(request,'myapp/predictdata.html')