# Exploring-Speaker-Recognition-Techniques

## Introduction
In the realm of speech processing and recognition, accurately identifying speakers from audio recordings is a crucial task with applications ranging from security authentication to personalized user experiences.
Our focus lies on implementing and comparing various state-of-the-art models, including Gaussian Mixture Models (GMMs), Recurrent Neural Networks (RNNs), and Multi Layer Perceptron (MLP), to discern the speaker's identity from audio samples. 
The dataset under examination comprises speeches delivered by five prominent leaders: Benjamin Netanyahu, Jens Stoltenberg, Julia Gillard, Margaret Thatcher, and Nelson Mandela. Each audio in the folder is a one-second 16000 sample rate PCM encoded. Our aim is to classify a given speech sample from the test dataset as one of the given classes. 
Throughout this project, we meticulously undertake feature extraction, model training, and evaluation (explained in the methodology and code) using a subset of the provided dataset. Our objective is to develop robust and efficient solutions for speaker recognition, aiming to attain high test accuracies that validate the efficacy of the employed methodologies

## Methodologies

This [notebook](https://github.com/ndhruv03/Exploring-Speaker-Recognition-Techniques/blob/main/speakerrecognition-combined-fa30a0.ipynb) has the code for all the following methodologies mentioned
### Gausian Mixture Models
Gaussian Mixture Models (GMMs) are probabilistic models commonly used for clustering and density estimation tasks. In the context of speaker recognition, GMMs can be employed to model the distribution of speech features extracted from different speakers. Each speaker's speech characteristics can be represented as a mixture of Gaussian distributions, where each Gaussian component captures a particular aspect of the speaker's speech patterns.
In the following code, we extract the Mel-Frequency Cepstral Coefficients (MFCCs) from audio files and for each speaker, a separate GMM is trained using the extracted MFCC features. The GMMs are initialized with three Gaussian components per speaker and are trained using the GaussianMixture class from scikit-learn. The trained GMMs are utilized to predict the speaker labels for the test set. For each test file, the log likelihood of the features given each speaker's GMM is computed, and the speaker with the highest likelihood is selected as the predicted speaker.

### Recurrent Neural Networks
Recurrent Neural Networks (RNNs) stand as a powerful variant of artificial neural networks, uniquely designed to process sequential data with temporal dependencies. Unlike traditional feedforward networks, RNNs possess recurrent connections that allow them to retain memory of past inputs while processing current ones. This inherent memory mechanism enables RNNs to excel in tasks requiring sequential modeling, such as natural language processing, time series analysis, and speech recognition. 
In the following code, we extract the Mel-Frequency Cepstral Coefficients (MFCCs) from audio files. The data is split into training,validation and testing datasets. We define a simple RNN model (using LSTM (Long Short Term Memory) Cells) which is trained using the Adam optimizer and the sparse categorical cross entropy loss for 20 epochs. The metric for evaluation is accuracy. The model is evaluated on the test dataset.

### Multi Layer Perceptron
Multilayer Perceptron (MLP) models represent a potent class of artificial neural networks, adept at capturing intricate patterns within diverse datasets. Utilizing multiple layers of interconnected neurons, MLPs excel in learning complex relationships between input and output data. It is a deep learning model.
In the following code, we extract the Mel-Frequency Cepstral Coefficients (MFCCs) from audio files. The data is split into training and testing datasets. We define a simple MLP model which is trained using the Adam optimizer and the sparse categorical cross entropy loss for 50 epochs. The metric for evaluation is accuracy. The model is evaluated on the test dataset.

## Additional Experiments
We have tried to use other features such as energy and Short-Time Fourier Transform (STFT) as well. Energy as a feature resulted in low accuracies. STFT reported good accuracies which were comparable to that of MFCCs.
We have also trained a Convolutional Neural Network on the extracted MFCC features and achieved a high accuracy. We have also included visualization of the audio samples and other metrics in the codebase. The code for the additional experiments are available in the repository itself.

## Results
The test accuracies for various models are as follows:
- Gaussian Mixture Models: 98.2%
- Recurrent Neural Networks: 99.71%
- Multi Layer Perceptron: 99.67%

## Conclusion
In conclusion, this project has explored and implemented multiple models to achieve accurate speaker recognition from a dedicated dataset. By employing Gaussian Mixture Models (GMMs), Recurrent Neural Networks (RNNs), and Multilayer Perceptron (MLP) architectures, we've endeavored to capture the nuanced characteristics of speech patterns for effective speaker identification. Through meticulous preprocessing, feature extraction, and training on a subset of the provided dataset, each model has been fine-tuned to discern the distinct signatures of individual speakers. By evaluating the performance of each model, we've gained valuable insights into their respective strengths and limitations in handling the complexities inherent in speaker recognition tasks.

## References
- https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset
- https://www.geeksforgeeks.org/gaussian-mixture-model/
- https://www.ibm.com/topics/recurrent-neural-networks
- https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning
- https://www.coursera.org/learn/convolutional-neural-networks
