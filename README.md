# Face-recognition with MTCNN & Facenet

# Models are used:

- MTCNN: detect face
- Facenet: exact features from that face
- Product of the scalar 2 vectors of features exacted from 2 different faces:

![image](https://user-images.githubusercontent.com/69660620/116783049-d5a40c80-aab6-11eb-9ff6-7edbbfc3f7d5.png)

![image](https://user-images.githubusercontent.com/69660620/116783125-303d6880-aab7-11eb-947b-df5adcd35450.png)

=> The more tiny the angular between 2 different vectors is, the more similar to each others the two faces are and vice versa.

# Operation:
- FunctionFull.py is composed of all subprograms that I use throughout this project.
- Run new.py if you want to update new members with new ID.
- Run verify.py if you want to check whether members are valid.
