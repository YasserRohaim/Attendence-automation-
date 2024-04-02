# Import necessary libraries/modules

import os
import pickle
import cv2
import face_recognition
import numpy as np
from sklearn import svm

def extract_frames(video_path, output_path, num_frames):
    """
    Extract a specified number of frames from a video file and save them with a specific naming convention.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to the output directory.
        num_frames (int): Number of frames to extract from the video.

    Returns:
        None
    """
    # Get the file name without extension
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a directory with the same name as the video file (without extension)
    output_directory = os.path.join(output_path, file_name)
    os.makedirs(output_directory, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {file_name}")
    print(f"Total Frames: {total_frames}")
    print(f"FPS: {fps}")

    # Calculate the indices of frames to be extracted
    step = max(total_frames // num_frames, 1)
    frame_indices = list(range(0, total_frames, step))[:num_frames]

    # Extract frames
    extracted_frames = 0
    for count, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {idx}. Skipping.")
            continue

        # Save the frame
        frame_name = f"{file_name}_frame_{count+1}.jpg"
        frame_path = os.path.join(output_directory, frame_name)
        cv2.imwrite(frame_path, frame)
        extracted_frames += 1

    # Release the video capture object
    cap.release()

    print(f"\nFrames Extracted: {extracted_frames}")
    
def face_encoding(img_file_location: str) -> np.ndarray:
    """
    Given the file location of an image, this function utilizes the face_recognition
    library to find and return the facial encodings for faces detected in the image.

    Parameters:
    - img_file_location (str): The file path of the image to process.

    Returns:
    - np.ndarray: An array containing the face encodings for the detected faces in the image.
    """
    # Load the image using face_recognition library
    img = face_recognition.load_image_file(img_file_location)

    # Convert the image to RGB format (required by face_recognition library)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use face_recognition library to find face locations in the image
    img_facial_locations = face_recognition.face_locations(img)

    # Use face_recognition library to find face encodings based on the known face locations
    img_encoding = face_recognition.face_encodings(
        img, known_face_locations=img_facial_locations
    )[0]

    # Return the obtained face encoding
    return img_encoding


def face_name_encoding(img_data: list) -> tuple[list, list]:
    """
    Given a list of directories in the current directory, this function extracts
    the names of persons (assuming names of directories match persons' names) and
    finds faces in any available image files. It then extracts the face encodings
    and locations, associating them with the respective persons.

    Parameters:
    - img_data (list): A list of directory names containing images for each person.

    Returns:
    - tuple[list, list]: A tuple containing two lists:
        - A list of face encodings for all detected faces.
        - A list of corresponding names for the detected faces.
    """

    # Get the list of persons from the input parameter
    persons = img_data

    # Check if previous face encodings exist
    encodings_exist = False
    if os.path.isfile("face_encodings_data.dat"):
        encodings_exist = True

    # Initialize dictionaries to store face encodings and associated file names
    all_face_encodings = {}
    file_names = {}
    encodings = []
    names = []

    # If previous encodings exist, load them from files
    if encodings_exist:
        with open("face_encodings_data.dat", "rb") as data_file:
            all_face_encodings = pickle.load(data_file)

        # If encodings exist, it is safe to assume that file names data also exists
        with open("file_names_data.dat", "rb") as data_file:
            file_names = pickle.load(data_file)

    # Iterate through each person's directory
    for person in persons:
        # Get the list of image files in the person's directory
        person_imgs = os.listdir(f"{IMG_DATA_DIR}/{person}")

        # Iterate through each image file in the person's directory
        for person_img in person_imgs:
            # Skip system files like '.DS_Store'
            if person_img == '.DS_Store':
                continue

            # Extract the person's name from the directory name
            name = os.path.splitext(person)[0]

            # Check if the person's name is not in the existing face encodings
            if name not in np.array(list(all_face_encodings.keys())):
                # If the person is new, save their first image encodings and filename
                all_face_encodings[name] = face_encoding(
                    f"{IMG_DATA_DIR}/{person}/{person_img}"
                )
                file_names[name] = [person_img]
                print(f"New person, {name}, is added")

            else:
                # If the person already has encodings, look for additional images of the same person
                # Encode those faces and connect the new encodings to the person
                if person_img not in file_names[name]:
                    nth_img_encodings = face_encoding(
                        f"{IMG_DATA_DIR}/{person}/{person_img}"
                    )
                    np.concatenate([all_face_encodings[name], nth_img_encodings])
                    file_names[name].append(person_img)
                    print(f"New image {person_img} encodings added for {name}")

    # Save the updated face encodings and file names to files
    with open("face_encodings_data.dat", "wb") as data_file:
        pickle.dump(all_face_encodings, data_file)

    with open("file_names_data.dat", "wb") as data_file:
        pickle.dump(file_names, data_file)

    # Convert the dictionaries to NumPy arrays for convenient use
    encodings = np.array(list(all_face_encodings.values()))
    names = np.array(list(all_face_encodings.keys()))

    # Return a tuple containing the face encodings and corresponding names
    return (encodings, names)

directory_path = '/Users/zaghloul2012/Desktop/attendance/Dataset/'

# List all files in the directory
files = os.listdir(directory_path)

# Extract frames from videos for all the dataset

video_dir = '/Users/zaghloul2012/Desktop/attendance/Dataset/'
output_path = '/Users/zaghloul2012/Desktop/attendance/Data/'
num_frames = 15

for file in files:
    if file == '.DS_Store':
        continue
    # Join the directory path with the file name
    file_path = os.path.join(directory_path, file)
    extract_frames(file_path, output_path, num_frames)
    
IMG_DATA_DIR = "./Data"

# We'll check if the Data directory exists or not
if not os.path.isdir(IMG_DATA_DIR):
    os.mkdir(IMG_DATA_DIR)  # and create one if it does not
    
img_data_obj = os.listdir(IMG_DATA_DIR)

# Check if there are any image directories
if len(img_data_obj) == 0:
    print("We need images of people to get work done")
else:
    # If there are image directories, obtain face encodings and names
    encodings, names = face_name_encoding(img_data_obj)
    
# Using SVM to fit the training data to train our model
clf = svm.SVC(kernel='rbf', gamma="scale", probability=True)
clf.fit(encodings, names)
