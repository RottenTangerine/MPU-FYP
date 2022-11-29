Hello professors, here is my progress report of my final year project, the intelligent scoring system. 


## Introduction (1min 30s)

Regarding the scoring system, there are already some applications. One of them is machine readable answer sheet, it is based on locating points to locate the position of the student's answer. Another one is online quiz system which requires the user to enter the answer in a specified area. However, both systems have shortcomings and there is room for improvement.

For machine readable answer sheets, it requires an answer sheet specifically designed for the test and may require a special optical recognition machine and students need to answer in the designated place. Another problem is that existing scoring system can only judge multiple-choice questions, neither fill-in-the-blank questions with standard answers nor questions that do not strictly matched to standard answers, such as reading comprehension and writing.

For the online quiz system, it requires participants to use electronic devices to access the system. Therefore, it is not practical to use it for schools that have strict rules about electronic devices. In addition, the cost of development and maintenance is very high. As a final point, it requires user to get used to typing on a keyboard, which is not friendly to old people.

## Description and Objectives (1min 50s)

Therefore, my final year project is to design an intelligent marking system employs deep learning technology in order to cut costs and increase usability. 

Students are permitted to answer examination questions on any white paper. The examiner then takes pictures of the answers and uploads them into the system. The system analyzes the student's answers and assigns a total grade based on their answers. 

The basic workflow of this project are shown in this picture. 

- First, the detector detects the position of the text and box them out.
- Then, The pixels containing the text are then cropped out using image processing techniques.
- Each image containing text is sent to a recognizer to recognize the text content and get a result that can be processed with machine encoding.
- Then we compare the results with the standard answers and totally them up, and get the final total score



the objective of this project is straight forward. ..

## Components

Next I will introduce two important components of this system completed during the semester: the detector and the recognizer.

## Conclusion (40s)

In this semester, I have completed the train dataset preprocessing, it is not covered in this presentation, but you can check it from my progress report, Besides, I also did the design and training of detection neural network based on CTPN model and the design and training of handwritten text recognition based on CRNN model.

However, throughout the project, there are still many unfinished parts. One of the main tasks is the design and training of the scoring model for judging text similarity based on Bert model. The another task is the design of the user interface.

That all of my progress report, thank you.