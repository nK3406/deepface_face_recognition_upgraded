# deepface_face_recognition_upgraded
Here is the face recognition system which use an upgraded version of deepface
I observed minor script deficiencies in Deepface's own open source code. 
First, it was not possible to access the terminal directly from the real-time face recognition function or the stream function in the common. 
I fixed this situation, 
now it is possible to activate the face recognition system by simply typing "--db_path [your own database path]" in the terminal. 
In addition, each time we call the files in the database on the same code, it was loading the model one by one. 
This meant a lot of time wasted for large datasets. 
To fix this, I made all this automatic and ready by printing the vector representations loaded into the model ready to the pickle file. 
The last thing I saw was the lack of script was the problem that brought too much to the camera. 
I fixed this problem, you can access the details from the source code.
I used Facenet512 model and opencv detector. you can change this also from terminal
How to use:
Write on terminal respectively: 
git clone https://github.com/nK3406/deepface_face_recognition_upgraded.git
cd deepface_face_recognition_upgraded
pip install -r requirements.txt
python realtime_face_recognition.py --db_path [your own database path]
Thats all :)
