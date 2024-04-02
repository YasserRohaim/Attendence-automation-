# Attendence-automation-
Face recognition for class attendence automation. The project works by using the open source model face recognition for generating embeddings of the faces in the database and storing them and then using an svm classifier to classify the embeddings of new images to a person in the database

## requirements
`pip install requirements.txt


## usage
after aquiring the database of the people you want to recognise use train_model.py to generate embeddings and train the model and then use test_model.py for usage

